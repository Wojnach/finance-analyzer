Reading additional input from stdin...
OpenAI Codex v0.120.0 (research preview)
--------
workdir: Q:\fa-adv-2026-05-11
model: gpt-5.4
provider: openai
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019e17a7-4a60-7cd3-83a1-bdf2c56a0936
--------
user
You are doing an ADVERSARIAL code review of the infrastructure subsystem (atomic I/O, locking, subprocess utils, dashboard, scheduled-task scripts) of a quantitative trading system at Q:\finance-analyzer. Sandbox: read-only.

In-scope files:
- portfolio/file_utils.py
- portfolio/shared_state.py
- portfolio/process_lock.py
- portfolio/subprocess_utils.py
- portfolio/config_validator.py
- portfolio/notification_text.py
- portfolio/message_store.py
- portfolio/shadow_registry.py
- portfolio/vector_memory.py
- portfolio/backtester.py
- dashboard/app.py
- dashboard/auth.py
- dashboard/export_static.py
- dashboard/house_blueprint.py
- dashboard/system_status.py
- dashboard/trading_status.py
- scripts/check_critical_errors.py
- scripts/fix_agent_dispatcher.py
- scripts/win/*.bat
- scripts/win/*.ps1
- conftest.py

Project rules:
- file_utils is the canonical atomic I/O layer. Raw json.loads(open(...).read()) is a defect anywhere in the codebase.
- atomic_append_jsonl must be safe against concurrent appenders + log rotation (the rotation race was just fixed in commit 3b623129).
- Dashboard auth: ?token=<dashboard_token> once, then 1-year rolling cookie. Bearer header for CLI clients. Cloudflare Access header bypasses local auth. NEVER expose API keys.
- PF-FixAgentDispatcher: per-category cooldown + exponential backoff (30m → 2h → 12h → effectively disabled). Tool allow-list = Read,Edit,Bash (no commit/push). `data/fix_agent.disabled` flag-file disables it entirely.
- Scheduled tasks must `set CLAUDECODE=` (the CLAUDECODE inheritance bug caused a 34h outage Feb 18-19).
- Singleton process lock: bat wrapper exits if schtasks restart hits the lock — kill orphans first.

Adversarial focus:
1. file_utils.atomic_write_json: race when two processes write the same path; tmp file leak on crash; encoding mismatch.
2. atomic_append_jsonl + log rotation race repeat (was just patched).
3. process_lock: stale lockfile (PID reused after reboot), non-cleanup on SIGKILL.
4. config_validator: schema drift; loading config.json (symlink to external file with API keys) — never log it.
5. Dashboard auth: token reuse / replay; cookie set with secure/HttpOnly flags; Cloudflare header trust chain.
6. Dashboard endpoints (33 of them): any that mutate state via GET; any that don't atomic-read.
7. subprocess_utils: shell-injection from user-derived args; missing universal_newlines / encoding=utf-8; orphan child on parent SIGTERM.
8. message_store: PII / secret persistence (Telegram tokens, BankID).
9. fix_agent_dispatcher: backoff state corruption; spawns even when fix_agent.disabled present; tool-allowlist drift.
10. check_critical_errors: TZ comparison bug on rolling 7d window.
11. PowerShell scripts: ExecutionPolicy bypass, hardcoded paths, missing -ErrorAction.
12. BAT scripts: CLAUDECODE not cleared before claude -p call.
13. conftest.py: fixtures that mutate global state without yield/cleanup.
14. vector_memory: embedding vector dim mismatch; unbounded growth.
15. shadow_registry: registry write race when multiple producers.

Output format — strict markdown:
# Codex adversarial review: infrastructure
## Summary
## P0 — Blockers
## P1 — High
## P2 — Medium
## P3 — Low
## Tests missing

Be concrete. Cite line numbers. No false-positive padding.
codex
Reviewing the infrastructure slice in `Q:\finance-analyzer` with the adversarial checklist you gave. I’m starting by reading the in-scope files and tracing the atomic I/O, locking, auth, subprocess, and scheduled-task paths before I rank findings.
  → Inspect in-scope files and map relevant code paths
  • Validate each adversarial focus area against concrete implementations
  • Assemble prioritized findings with exact file/line citations and testing gaps
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg --files portfolio dashboard scripts conftest.py' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git status --short' in Q:\finance-analyzer
2026-05-11T15:28:35.381394Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git rev-parse --show-toplevel'` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git rev-parse --show-toplevel' in Q:\finance-analyzer
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'git rev-parse --show-toplevel'` rejected: blocked by policy
2026-05-11T15:28:36.221634Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
scripts\auto-improve.bat
portfolio\accuracy_stats.py
portfolio\accuracy_degradation.py
scripts\auto-improve-codex.bat
scripts\audit_accuracy_drops.py
portfolio\agent_invocation.py
scripts\after-hours-research.bat
scripts\avanza_metals_ladder.py
scripts\avanza_metals_check.py
dashboard\export_static.py
scripts\avanza_login.py
dashboard\auth.py
dashboard\app.py
scripts\avanza_orders.py
scripts\avanza_smoke_test.py
dashboard\system_status.py
dashboard\house_blueprint.py
scripts\backfill_accuracy_snapshots.py
scripts\backfill_llm_outcomes.py
scripts\backfill_sentiment_shadow.py
portfolio\analyze.py
portfolio\alpha_vantage.py
portfolio\alert_budget.py
portfolio\autonomous.py
portfolio\api_utils.py
dashboard\trading_status.py
portfolio\avanza_client.py
portfolio\avanza_account_check.py
portfolio\avanza_control.py
scripts\backtest_new_signals.py
scripts\backtest_drift_regime_gate.py
portfolio\avanza_orders.py
portfolio\avanza_order_lock.py
scripts\backtest_pair_reversion.py
scripts\backtest_realized_skewness.py
scripts\benchmark_gpu_models.py
portfolio\bert_sentiment.py
portfolio\backtester.py
portfolio\avanza_tracker.py
portfolio\avanza_session.py
portfolio\avanza_resilient_page.py
portfolio\claude_gate.py
portfolio\circuit_breaker.py
portfolio\bigbet.py
portfolio\config_validator.py
portfolio\cost_model.py
scripts\check_critical_errors.py
portfolio\crypto_macro_data.py
scripts\fin_snipe.py
scripts\fin_fish_monitor.py
scripts\fin_fish.py
scripts\elongir.bat
scripts\download_models.py
scripts\cloudflared-config.yml
scripts\cleanup_settings_20260508.sh
scripts\fix_agent_dispatcher.py
scripts\fish_straddle.py
scripts\fish_preflight.py
scripts\fish_monitor_live.py
scripts\fin_snipe_manager.py
scripts\ft-download-data.sh
scripts\ft-backtest.sh
scripts\fix_cloudflared_imagepath.ps1
scripts\ft-dry-run.sh
scripts\ft-health.sh
dashboard\static\index_legacy.html
dashboard\static\index.html
dashboard\static\sw.js
dashboard\static\manifest.webmanifest
portfolio\avanza\types.py
portfolio\avanza\trading.py
portfolio\avanza\tick_rules.py
portfolio\avanza\streaming.py
portfolio\avanza\search.py
portfolio\avanza\scanner.py
portfolio\avanza\market_data.py
portfolio\avanza\client.py
portfolio\avanza\auth.py
portfolio\avanza\account.py
dashboard\static\icons\icon-192.png
dashboard\static\icons\apple-touch-icon-180.png
dashboard\static\icons\icon-512.png
dashboard\static\icons\icon-512-maskable.png
scripts\iskbet.py
scripts\health_check.py
scripts\grid_fisher_probe.py
scripts\golddigger_backtest_today.py
scripts\golddigger.bat
scripts\ft.sh
scripts\ft-walkforward.sh
scripts\ft-walkforward.py
scripts\ft-test.sh
scripts\ft-hyperopt.sh
scripts\mstr_loop_backtest.py
scripts\monitor_silver_exit.py
scripts\lora_backtest.py
scripts\loop_health_watchdog.py
scripts\loop_health_report.py
scripts\iskbet_replay.py
dashboard\static\css\tokens.css
dashboard\static\css\responsive.css
dashboard\static\css\layout.css
dashboard\static\css\components.css
dashboard\static\css\base.css
scripts\oil_loop_scorecard.py
scripts\mstr_loop_scorecard.py
scripts\perf\backtest_conf_threshold.py
scripts\perf\profile_utility_overlay.py
scripts\run_mutation_test.py
scripts\review_shadow_signals.py
scripts\restart_loop.py
scripts\resolve_loop_audit_errors.py
scripts\replay_consensus.py
scripts\probe_oil_warrants.py
scripts\prepare_kronos_training_data.py
scripts\pf.py
scripts\pf-sync.bat
scripts\pf-silver-orb.bat
scripts\pf-crypto.bat
scripts\signal_correlation_audit.py
scripts\signal_backtest.py
scripts\signal-research.bat
scripts\setup_wsl_claude.ps1
scripts\setup_tunnel.bat
scripts\setup_gh_pages.py
scripts\sync_dashboard.bat
scripts\signal_research_phase34.py
scripts\signal_research_extract.py
scripts\sysmon.py
scripts\sync_dashboard.py
scripts\trigger_simulation.py
dashboard\static\js\polling.js
dashboard\static\js\main.js
dashboard\static\js\format.js
dashboard\static\js\fetch.js
dashboard\static\js\desktop-mode.js
scripts\tune_new_signals.py
scripts\verify_kronos.py
dashboard\static\js\theme.js
dashboard\static\js\state.js
dashboard\static\js\router.js
scripts\write_research_outputs.py
dashboard\static\js\views\signals.js
dashboard\static\js\views\settings.js
dashboard\static\js\views\prices.js
dashboard\static\js\views\portfolio.js
dashboard\static\js\views\more.js
dashboard\static\js\views\metals.js
dashboard\static\js\views\messages.js
dashboard\static\js\views\home.js
dashboard\static\js\views\health.js
dashboard\static\js\views\golddigger.js
dashboard\static\js\views\equity.js
dashboard\static\js\views\decisions.js
dashboard\static\js\views\decision-detail.js
dashboard\static\js\views\avanza.js
dashboard\static\js\views\assets.js
portfolio\gold_precompute.py
dashboard\static\js\render\trading-status-card.js
dashboard\static\js\render\system-status-hero.js
dashboard\static\js\render\signals-heatmap.js
dashboard\static\js\render\signal-pulse-card.js
dashboard\static\js\render\llm-inference-card.js
dashboard\static\js\render\layer2-activity-card.js
dashboard\static\js\render\errors-panel.js
dashboard\static\js\render\accuracy.js
scripts\verify_tunnel.py
scripts\verify_tunnel_alerted.py
portfolio\fear_greed.py
portfolio\exposure_coach.py
portfolio\exit_optimizer.py
portfolio\equity_curve.py
dashboard\static\js\components\signal-row.js
dashboard\static\js\components\pulse-dot.js
dashboard\static\js\components\position-card.js
dashboard\static\js\components\pnl-card.js
dashboard\static\js\components\mini-chart.js
dashboard\static\js\components\filter-chip.js
dashboard\static\js\components\error-banner.js
dashboard\static\js\components\empty-state.js
dashboard\static\js\components\decision-card.js
dashboard\static\js\components\consensus-chip.js
dashboard\static\js\components\bottom-sheet.js
dashboard\static\js\components\accordion.js
portfolio\golddigger\state.py
portfolio\golddigger\signal.py
portfolio\golddigger\runner.py
portfolio\golddigger\risk.py
portfolio\golddigger\data_provider.py
portfolio\golddigger\config.py
portfolio\golddigger\bot.py
portfolio\golddigger\augmented_signals.py
dashboard\static\js\charts\mini-sparkline.js
dashboard\static\js\charts\equity-chart.js
dashboard\static\js\charts\chart-config.js
dashboard\static\js\charts\accuracy-chart.js
portfolio\fx_rates.py
portfolio\futures_data.py
portfolio\funding_rate.py
portfolio\forecast_signal.py
portfolio\forecast_accuracy.py
portfolio\fomc_dates.py
portfolio\focus_analysis.py
portfolio\fish_monitor_smart.py
portfolio\fish_instrument_finder.py
portfolio\fin_snipe_manager.py
portfolio\fin_snipe.py
portfolio\fin_fish.py
portfolio\fin_evolve.py
portfolio\file_utils.py
portfolio\feature_normalizer.py
portfolio\message_store.py
portfolio\memory_consolidation.py
portfolio\market_timing.py
portfolio\market_health.py
portfolio\main.py
portfolio\macro_context.py
portfolio\loop_health.py
portfolio\data_collector.py
portfolio\loop_contract.py
portfolio\daily_digest.py
portfolio\log_rotation.py
portfolio\cumulative_tracker.py
portfolio\logging_config.py
portfolio\crypto_scheduler.py
portfolio\local_llm_report.py
portfolio\crypto_precompute.py
portfolio\llm_probability_log.py
portfolio\llm_prewarmer.py
portfolio\llm_outcome_backfill.py
portfolio\digest.py
portfolio\llm_calibration.py
portfolio\decision_outcome_tracker.py
portfolio\llm_batch.py
portfolio\data_refresh.py
portfolio\llama_server.py
portfolio\linear_factor.py
portfolio\earnings_calendar.py
portfolio\kelly_sizing.py
portfolio\econ_dates.py
portfolio\kelly_metals.py
portfolio\journal_index.py
portfolio\journal.py
portfolio\iskbets.py
portfolio\instrument_profile.py
scripts\win\install-health-check-tasks.ps1
portfolio\indicators.py
scripts\win\install-golddigger-task.ps1
scripts\win\install-fix-agent-task.ps1
scripts\win\install-crypto-loop-task.ps1
scripts\win\install-adversarial-review-task.ps1
scripts\win\golddigger.bat
scripts\win\golddigger-loop.bat
scripts\win\ft-test.bat
scripts\win\ft-hyperopt.bat
scripts\win\ft-dry-run.bat
scripts\win\ft-download-data.bat
scripts\win\ft-backtest.bat
scripts\win\fin-snipe.bat
scripts\win\fin-snipe-manager.bat
scripts\win\crypto-loop.bat
scripts\win\adversarial-review.bat
scripts\win\add-cloudflared-path.ps1
scripts\win\install-meta-learner-task.ps1
scripts\win\install-market-tasks.ps1
scripts\win\install-loop-resume-task.ps1
scripts\win\install-loop-health-watchdog-task.ps1
scripts\win\install-loop-health-report-task.ps1
scripts\win\install-loop-health-daily-task.ps1
scripts\win\install-log-rotate-task.ps1
scripts\win\install-local-llm-report-task.ps1
scripts\win\install-rc-keepalive-task.ps1
scripts\win\install-oil-loop-task.ps1
scripts\win\install-mstr-loop-task.ps1
scripts\win\install-metals-loop-task.ps1
scripts\win\install-rc-watchdog-task.ps1
scripts\win\install-rc-server-task.ps1
scripts\win\install-research-task.ps1
portfolio\ic_computation.py
portfolio\http_retry.py
portfolio\health.py
portfolio\grid_tiers.py
portfolio\grid_fisher_config.py
portfolio\grid_fisher.py
portfolio\gpu_gate.py
portfolio\price_source.py
portfolio\portfolio_validator.py
portfolio\portfolio_mgr.py
portfolio\perception_gate.py
portfolio\outcome_tracker.py
scripts\win\install-signal-research-task.ps1
portfolio\orb_predictor.py
portfolio\orb_postmortem.py
portfolio\orb_backtest.py
portfolio\onchain_data.py
portfolio\oil_precompute.py
portfolio\oil_grid_signal.py
portfolio\ml_signal.py
portfolio\notification_text.py
portfolio\ministral_trader.py
portfolio\news_keywords.py
portfolio\ministral_signal.py
portfolio\microstructure_state.py
portfolio\multi_agent_layer2.py
portfolio\microstructure.py
portfolio\mstr_precompute.py
portfolio\meta_learner.py
portfolio\metals_precompute.py
portfolio\metals_orderbook.py
portfolio\metals_ladder.py
portfolio\metals_cross_assets.py
portfolio\message_throttle.py
portfolio\monte_carlo.py
portfolio\ml_trainer.py
portfolio\monte_carlo_risk.py
portfolio\shared_state.py
portfolio\shadow_registry.py
portfolio\session_calendar.py
portfolio\sentiment_shadow_backfill.py
portfolio\sentiment.py
portfolio\seasonality_updater.py
portfolio\seasonality.py
portfolio\risk_management.py
portfolio\reporting.py
portfolio\regime_alerts.py
portfolio\reflection.py
portfolio\qwen3_trader.py
portfolio\qwen3_signal.py
portfolio\prophecy.py
portfolio\process_lock.py
portfolio\price_targets.py
portfolio\signal_registry.py
portfolio\signal_postmortem.py
portfolio\signal_history.py
portfolio\signal_engine.py
portfolio\signal_decay_alert.py
portfolio\signal_db.py
scripts\win\train-after-hours.bat
scripts\win\silver-monitor.bat
scripts\win\settings-cleanup-20260508.bat
scripts\win\rc-watchdog.ps1
scripts\win\rc-server.bat
scripts\win\rc-server-ensure.ps1
scripts\win\rc-server-3.bat
scripts\win\rc-server-2.bat
scripts\win\rc-keepalive.ps1
scripts\win\pf.bat
scripts\win\pf-shadow-review.bat
scripts\win\pf-restart.ps1
scripts\win\pf-restart.bat
scripts\win\pf-outcome-check.bat
scripts\win\pf-loop.bat
scripts\win\pf-loop-ensure.ps1
scripts\win\pf-local-llm-report.bat
scripts\win\pf-llm-backfill.bat
scripts\win\pf-agent.bat
scripts\win\oil-loop.bat
scripts\win\mstr-loop.bat
scripts\win\metals-loop.bat
scripts\win\metals-arm-stop-once.bat
scripts\win\meta-learner-retrain.bat
portfolio\weekly_digest.py
portfolio\warrant_portfolio.py
portfolio\vector_memory.py
portfolio\trigger.py
portfolio\train_signal_weights.py
portfolio\trade_validation.py
portfolio\trade_risk_classifier.py
portfolio\trade_guards.py
portfolio\tinylora_trainer.py
portfolio\ticker_accuracy.py
portfolio\tickers.py
portfolio\telegram_poller.py
portfolio\telegram_notifications.py
portfolio\subprocess_utils.py
portfolio\mstr_loop\config.py
portfolio\signal_weight_optimizer.py
portfolio\signal_weights.py
portfolio\signal_utils.py
portfolio\signal_state_since.py
portfolio\social_sentiment.py
portfolio\silver_precompute.py
portfolio\stats.py
portfolio\short_horizon.py
portfolio\mstr_loop\data_provider.py
portfolio\elongir\bot.py
portfolio\elongir\config.py
portfolio\mstr_loop\telegram_report.py
portfolio\elongir\state.py
portfolio\elongir\signal.py
portfolio\elongir\runner.py
portfolio\elongir\risk.py
portfolio\elongir\indicators.py
portfolio\elongir\data_provider.py
portfolio\mstr_loop\risk.py
portfolio\mstr_loop\loop.py
portfolio\mstr_loop\execution.py
portfolio\mstr_loop\session.py
portfolio\mstr_loop\state.py
portfolio\signals\calendar_seasonal.py
portfolio\signals\candlestick.py
portfolio\strategies\orchestrator.py
portfolio\strategies\golddigger_strategy.py
portfolio\strategies\elongir_strategy.py
portfolio\strategies\base.py
portfolio\signals\crypto_macro.py
portfolio\signals\crypto_evrp.py
portfolio\signals\cross_asset_tsmom.py
portfolio\signals\credit_spread.py
portfolio\signals\cot_positioning.py
portfolio\signals\copper_gold_ratio.py
portfolio\signals\complexity_gap_regime.py
portfolio\signals\claude_fundamental.py
portfolio\signals\fibonacci.py
portfolio\signals\econ_calendar.py
portfolio\signals\dxy_cross_asset.py
portfolio\signals\drift_regime_gate.py
portfolio\signals\cubic_trend_persistence.py
portfolio\signals\futures_basis.py
portfolio\signals\forecast.py
portfolio\signals\futures_flow.py
portfolio\signals\gold_real_yield_paradox.py
portfolio\mstr_loop\strategies\premium_arb.py
portfolio\mstr_loop\strategies\overnight_gap.py
portfolio\mstr_loop\strategies\momentum_rider.py
portfolio\mstr_loop\strategies\mean_reversion.py
portfolio\mstr_loop\strategies\earnings_play.py
portfolio\mstr_loop\strategies\base.py
portfolio\signals\metals_cross_asset.py
portfolio\signals\mean_reversion.py
portfolio\signals\mahalanobis_turbulence.py
portfolio\signals\macro_regime.py
portfolio\signals\intraday_seasonality.py
portfolio\signals\hurst_regime.py
portfolio\signals\heikin_ashi.py
portfolio\signals\hash_ribbons.py
portfolio\signals\orderbook_flow.py
portfolio\signals\news_event.py
portfolio\signals\network_momentum.py
portfolio\signals\momentum_factors.py
portfolio\signals\momentum.py
portfolio\signals\ovx_metals_spillover.py
portfolio\signals\oscillators.py
portfolio\signals\realized_skewness.py
portfolio\signals\residual_pair_reversion.py
portfolio\signals\xtrend_equity_spillover.py
portfolio\signals\williams_vix_fix.py
portfolio\signals\vwap_zscore_mr.py
portfolio\signals\vol_ratio_regime.py
portfolio\signals\volume_flow.py
portfolio\signals\volatility.py
portfolio\signals\vix_term_structure.py
portfolio\signals\trend.py
portfolio\signals\treasury_risk_rotation.py
portfolio\signals\structure.py
portfolio\signals\statistical_jump_regime.py
portfolio\signals\smart_money.py
portfolio\signals\shannon_entropy.py
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: conftest.py: The system cannot find the file specified. (os error 2)

 exited 1 in 840ms:
scripts\auto-improve.bat
portfolio\accuracy_stats.py
portfolio\accuracy_degradation.py
scripts\auto-improve-codex.bat
scripts\audit_accuracy_drops.py
portfolio\agent_invocation.py
scripts\after-hours-research.bat
scripts\avanza_metals_ladder.py
scripts\avanza_metals_check.py
dashboard\export_static.py
scripts\avanza_login.py
dashboard\auth.py
dashboard\app.py
scripts\avanza_orders.py
scripts\avanza_smoke_test.py
dashboard\system_status.py
dashboard\house_blueprint.py
scripts\backfill_accuracy_snapshots.py
scripts\backfill_llm_outcomes.py
scripts\backfill_sentiment_shadow.py
portfolio\analyze.py
portfolio\alpha_vantage.py
portfolio\alert_budget.py
portfolio\autonomous.py
portfolio\api_utils.py
dashboard\trading_status.py
portfolio\avanza_client.py
portfolio\avanza_account_check.py
portfolio\avanza_control.py
scripts\backtest_new_signals.py
scripts\backtest_drift_regime_gate.py
portfolio\avanza_orders.py
portfolio\avanza_order_lock.py
scripts\backtest_pair_reversion.py
scripts\backtest_realized_skewness.py
scripts\benchmark_gpu_models.py
portfolio\bert_sentiment.py
portfolio\backtester.py
portfolio\avanza_tracker.py
portfolio\avanza_session.py
portfolio\avanza_resilient_page.py
portfolio\claude_gate.py
portfolio\circuit_breaker.py
portfolio\bigbet.py
portfolio\config_validator.py
portfolio\cost_model.py
scripts\check_critical_errors.py
portfolio\crypto_macro_data.py
scripts\fin_snipe.py
scripts\fin_fish_monitor.py
scripts\fin_fish.py
scripts\elongir.bat
scripts\download_models.py
scripts\cloudflared-config.yml
scripts\cleanup_settings_20260508.sh
scripts\fix_agent_dispatcher.py
scripts\fish_straddle.py
scripts\fish_preflight.py
scripts\fish_monitor_live.py
scripts\fin_snipe_manager.py
scripts\ft-download-data.sh
scripts\ft-backtest.sh
scripts\fix_cloudflared_imagepath.ps1
scripts\ft-dry-run.sh
scripts\ft-health.sh
dashboard\static\index_legacy.html
dashboard\static\index.html
dashboard\static\sw.js
dashboard\static\manifest.webmanifest
portfolio\avanza\types.py
portfolio\avanza\trading.py
portfolio\avanza\tick_rules.py
portfolio\avanza\streaming.py
portfolio\avanza\search.py
portfolio\avanza\scanner.py
portfolio\avanza\market_data.py
portfolio\avanza\client.py
portfolio\avanza\auth.py
portfolio\avanza\account.py
dashboard\static\icons\icon-192.png
dashboard\static\icons\apple-touch-icon-180.png
dashboard\static\icons\icon-512.png
dashboard\static\icons\icon-512-maskable.png
scripts\iskbet.py
scripts\health_check.py
scripts\grid_fisher_probe.py
scripts\golddigger_backtest_today.py
scripts\golddigger.bat
scripts\ft.sh
scripts\ft-walkforward.sh
scripts\ft-walkforward.py
scripts\ft-test.sh
scripts\ft-hyperopt.sh
scripts\mstr_loop_backtest.py
scripts\monitor_silver_exit.py
scripts\lora_backtest.py
scripts\loop_health_watchdog.py
scripts\loop_health_report.py
scripts\iskbet_replay.py
dashboard\static\css\tokens.css
dashboard\static\css\responsive.css
dashboard\static\css\layout.css
dashboard\static\css\components.css
dashboard\static\css\base.css
scripts\oil_loop_scorecard.py
scripts\mstr_loop_scorecard.py
scripts\perf\backtest_conf_threshold.py
scripts\perf\profile_utility_overlay.py
scripts\run_mutation_test.py
scripts\review_shadow_signals.py
scripts\restart_loop.py
scripts\resolve_loop_audit_errors.py
scripts\replay_consensus.py
scripts\probe_oil_warrants.py
scripts\prepare_kronos_training_data.py
scripts\pf.py
scripts\pf-sync.bat
scripts\pf-silver-orb.bat
scripts\pf-crypto.bat
scripts\signal_correlation_audit.py
scripts\signal_backtest.py
scripts\signal-research.bat
scripts\setup_wsl_claude.ps1
scripts\setup_tunnel.bat
scripts\setup_gh_pages.py
scripts\sync_dashboard.bat
scripts\signal_research_phase34.py
scripts\signal_research_extract.py
scripts\sysmon.py
scripts\sync_dashboard.py
scripts\trigger_simulation.py
dashboard\static\js\polling.js
dashboard\static\js\main.js
dashboard\static\js\format.js
dashboard\static\js\fetch.js
dashboard\static\js\desktop-mode.js
scripts\tune_new_signals.py
scripts\verify_kronos.py
dashboard\static\js\theme.js
dashboard\static\js\state.js
dashboard\static\js\router.js
scripts\write_research_outputs.py
dashboard\static\js\views\signals.js
dashboard\static\js\views\settings.js
dashboard\static\js\views\prices.js
dashboard\static\js\views\portfolio.js
dashboard\static\js\views\more.js
dashboard\static\js\views\metals.js
dashboard\static\js\views\messages.js
dashboard\static\js\views\home.js
dashboard\static\js\views\health.js
dashboard\static\js\views\golddigger.js
dashboard\static\js\views\equity.js
dashboard\static\js\views\decisions.js
dashboard\static\js\views\decision-detail.js
dashboard\static\js\views\avanza.js
dashboard\static\js\views\assets.js
portfolio\gold_precompute.py
dashboard\static\js\render\trading-status-card.js
dashboard\static\js\render\system-status-hero.js
dashboard\static\js\render\signals-heatmap.js
dashboard\static\js\render\signal-pulse-card.js
dashboard\static\js\render\llm-inference-card.js
dashboard\static\js\render\layer2-activity-card.js
dashboard\static\js\render\errors-panel.js
dashboard\static\js\render\accuracy.js
scripts\verify_tunnel.py
scripts\verify_tunnel_alerted.py
portfolio\fear_greed.py
portfolio\exposure_coach.py
portfolio\exit_optimizer.py
portfolio\equity_curve.py
dashboard\static\js\components\signal-row.js
dashboard\static\js\components\pulse-dot.js
dashboard\static\js\components\position-card.js
dashboard\static\js\components\pnl-card.js
dashboard\static\js\components\mini-chart.js
dashboard\static\js\components\filter-chip.js
dashboard\static\js\components\error-banner.js
dashboard\static\js\components\empty-state.js
dashboard\static\js\components\decision-card.js
dashboard\static\js\components\consensus-chip.js
dashboard\static\js\components\bottom-sheet.js
dashboard\static\js\components\accordion.js
portfolio\golddigger\state.py
portfolio\golddigger\signal.py
portfolio\golddigger\runner.py
portfolio\golddigger\risk.py
portfolio\golddigger\data_provider.py
portfolio\golddigger\config.py
portfolio\golddigger\bot.py
portfolio\golddigger\augmented_signals.py
dashboard\static\js\charts\mini-sparkline.js
dashboard\static\js\charts\equity-chart.js
dashboard\static\js\charts\chart-config.js
dashboard\static\js\charts\accuracy-chart.js
portfolio\fx_rates.py
portfolio\futures_data.py
portfolio\funding_rate.py
portfolio\forecast_signal.py
portfolio\forecast_accuracy.py
portfolio\fomc_dates.py
portfolio\focus_analysis.py
portfolio\fish_monitor_smart.py
portfolio\fish_instrument_finder.py
portfolio\fin_snipe_manager.py
portfolio\fin_snipe.py
portfolio\fin_fish.py
portfolio\fin_evolve.py
portfolio\file_utils.py
portfolio\feature_normalizer.py
portfolio\message_store.py
portfolio\memory_consolidation.py
portfolio\market_timing.py
portfolio\market_health.py
portfolio\main.py
portfolio\macro_context.py
portfolio\loop_health.py
portfolio\data_collector.py
portfolio\loop_contract.py
portfolio\daily_digest.py
portfolio\log_rotation.py
portfolio\cumulative_tracker.py
portfolio\logging_config.py
portfolio\crypto_scheduler.py
portfolio\local_llm_report.py
portfolio\crypto_precompute.py
portfolio\llm_probability_log.py
portfolio\llm_prewarmer.py
portfolio\llm_outcome_backfill.py
portfolio\digest.py
portfolio\llm_calibration.py
portfolio\decision_outcome_tracker.py
portfolio\llm_batch.py
portfolio\data_refresh.py
portfolio\llama_server.py
portfolio\linear_factor.py
portfolio\earnings_calendar.py
portfolio\kelly_sizing.py
portfolio\econ_dates.py
portfolio\kelly_metals.py
portfolio\journal_index.py
portfolio\journal.py
portfolio\iskbets.py
portfolio\instrument_profile.py
scripts\win\install-health-check-tasks.ps1
portfolio\indicators.py
scripts\win\install-golddigger-task.ps1
scripts\win\install-fix-agent-task.ps1
scripts\win\install-crypto-loop-task.ps1
scripts\win\install-adversarial-review-task.ps1
scripts\win\golddigger.bat
scripts\win\golddigger-loop.bat
scripts\win\ft-test.bat
scripts\win\ft-hyperopt.bat
scripts\win\ft-dry-run.bat
scripts\win\ft-download-data.bat
scripts\win\ft-backtest.bat
scripts\win\fin-snipe.bat
scripts\win\fin-snipe-manager.bat
scripts\win\crypto-loop.bat
scripts\win\adversarial-review.bat
scripts\win\add-cloudflared-path.ps1
scripts\win\install-meta-learner-task.ps1
scripts\win\install-market-tasks.ps1
scripts\win\install-loop-resume-task.ps1
scripts\win\install-loop-health-watchdog-task.ps1
scripts\win\install-loop-health-report-task.ps1
scripts\win\install-loop-health-daily-task.ps1
scripts\win\install-log-rotate-task.ps1
scripts\win\install-local-llm-report-task.ps1
scripts\win\install-rc-keepalive-task.ps1
scripts\win\install-oil-loop-task.ps1
scripts\win\install-mstr-loop-task.ps1
scripts\win\install-metals-loop-task.ps1
scripts\win\install-rc-watchdog-task.ps1
scripts\win\install-rc-server-task.ps1
scripts\win\install-research-task.ps1
portfolio\ic_computation.py
portfolio\http_retry.py
portfolio\health.py
portfolio\grid_tiers.py
portfolio\grid_fisher_config.py
portfolio\grid_fisher.py
portfolio\gpu_gate.py
portfolio\price_source.py
portfolio\portfolio_validator.py
portfolio\portfolio_mgr.py
portfolio\perception_gate.py
portfolio\outcome_tracker.py
scripts\win\install-signal-research-task.ps1
portfolio\orb_predictor.py
portfolio\orb_postmortem.py
portfolio\orb_backtest.py
portfolio\onchain_data.py
portfolio\oil_precompute.py
portfolio\oil_grid_signal.py
portfolio\ml_signal.py
portfolio\notification_text.py
portfolio\ministral_trader.py
portfolio\news_keywords.py
portfolio\ministral_signal.py
portfolio\microstructure_state.py
portfolio\multi_agent_layer2.py
portfolio\microstructure.py
portfolio\mstr_precompute.py
portfolio\meta_learner.py
portfolio\metals_precompute.py
portfolio\metals_orderbook.py
portfolio\metals_ladder.py
portfolio\metals_cross_assets.py
portfolio\message_throttle.py
portfolio\monte_carlo.py
portfolio\ml_trainer.py
portfolio\monte_carlo_risk.py
portfolio\shared_state.py
portfolio\shadow_registry.py
portfolio\session_calendar.py
portfolio\sentiment_shadow_backfill.py
portfolio\sentiment.py
portfolio\seasonality_updater.py
portfolio\seasonality.py
portfolio\risk_management.py
portfolio\reporting.py
portfolio\regime_alerts.py
portfolio\reflection.py
portfolio\qwen3_trader.py
portfolio\qwen3_signal.py
portfolio\prophecy.py
portfolio\process_lock.py
portfolio\price_targets.py
portfolio\signal_registry.py
portfolio\signal_postmortem.py
portfolio\signal_history.py
portfolio\signal_engine.py
portfolio\signal_decay_alert.py
portfolio\signal_db.py
scripts\win\train-after-hours.bat
scripts\win\silver-monitor.bat
scripts\win\settings-cleanup-20260508.bat
scripts\win\rc-watchdog.ps1
scripts\win\rc-server.bat
scripts\win\rc-server-ensure.ps1
scripts\win\rc-server-3.bat
scripts\win\rc-server-2.bat
scripts\win\rc-keepalive.ps1
scripts\win\pf.bat
scripts\win\pf-shadow-review.bat
scripts\win\pf-restart.ps1
scripts\win\pf-restart.bat
scripts\win\pf-outcome-check.bat
scripts\win\pf-loop.bat
scripts\win\pf-loop-ensure.ps1
scripts\win\pf-local-llm-report.bat
scripts\win\pf-llm-backfill.bat
scripts\win\pf-agent.bat
scripts\win\oil-loop.bat
scripts\win\mstr-loop.bat
scripts\win\metals-loop.bat
scripts\win\metals-arm-stop-once.bat
scripts\win\meta-learner-retrain.bat
portfolio\weekly_digest.py
portfolio\warrant_portfolio.py
portfolio\vector_memory.py
portfolio\trigger.py
portfolio\train_signal_weights.py
portfolio\trade_validation.py
portfolio\trade_risk_classifier.py
portfolio\trade_guards.py
portfolio\tinylora_trainer.py
portfolio\ticker_accuracy.py
portfolio\tickers.py
portfolio\telegram_poller.py
portfolio\telegram_notifications.py
portfolio\subprocess_utils.py
portfolio\mstr_loop\config.py
portfolio\signal_weight_optimizer.py
portfolio\signal_weights.py
portfolio\signal_utils.py
portfolio\signal_state_since.py
portfolio\social_sentiment.py
portfolio\silver_precompute.py
portfolio\stats.py
portfolio\short_horizon.py
portfolio\mstr_loop\data_provider.py
portfolio\elongir\bot.py
portfolio\elongir\config.py
portfolio\mstr_loop\telegram_report.py
portfolio\elongir\state.py
portfolio\elongir\signal.py
portfolio\elongir\runner.py
portfolio\elongir\risk.py
portfolio\elongir\indicators.py
portfolio\elongir\data_provider.py
portfolio\mstr_loop\risk.py
portfolio\mstr_loop\loop.py
portfolio\mstr_loop\execution.py
portfolio\mstr_loop\session.py
portfolio\mstr_loop\state.py
portfolio\signals\calendar_seasonal.py
portfolio\signals\candlestick.py
portfolio\strategies\orchestrator.py
portfolio\strategies\golddigger_strategy.py
portfolio\strategies\elongir_strategy.py
portfolio\strategies\base.py
portfolio\signals\crypto_macro.py
portfolio\signals\crypto_evrp.py
portfolio\signals\cross_asset_tsmom.py
portfolio\signals\credit_spread.py
portfolio\signals\cot_positioning.py
portfolio\signals\copper_gold_ratio.py
portfolio\signals\complexity_gap_regime.py
portfolio\signals\claude_fundamental.py
portfolio\signals\fibonacci.py
portfolio\signals\econ_calendar.py
portfolio\signals\dxy_cross_asset.py
portfolio\signals\drift_regime_gate.py
portfolio\signals\cubic_trend_persistence.py
portfolio\signals\futures_basis.py
portfolio\signals\forecast.py
portfolio\signals\futures_flow.py
portfolio\signals\gold_real_yield_paradox.py
portfolio\mstr_loop\strategies\premium_arb.py
portfolio\mstr_loop\strategies\overnight_gap.py
portfolio\mstr_loop\strategies\momentum_rider.py
portfolio\mstr_loop\strategies\mean_reversion.py
portfolio\mstr_loop\strategies\earnings_play.py
portfolio\mstr_loop\strategies\base.py
portfolio\signals\metals_cross_asset.py
portfolio\signals\mean_reversion.py
portfolio\signals\mahalanobis_turbulence.py
portfolio\signals\macro_regime.py
portfolio\signals\intraday_seasonality.py
portfolio\signals\hurst_regime.py
portfolio\signals\heikin_ashi.py
portfolio\signals\hash_ribbons.py
portfolio\signals\orderbook_flow.py
portfolio\signals\news_event.py
portfolio\signals\network_momentum.py
portfolio\signals\momentum_factors.py
portfolio\signals\momentum.py
portfolio\signals\ovx_metals_spillover.py
portfolio\signals\oscillators.py
portfolio\signals\realized_skewness.py
portfolio\signals\residual_pair_reversion.py
portfolio\signals\xtrend_equity_spillover.py
portfolio\signals\williams_vix_fix.py
portfolio\signals\vwap_zscore_mr.py
portfolio\signals\vol_ratio_regime.py
portfolio\signals\volume_flow.py
portfolio\signals\volatility.py
portfolio\signals\vix_term_structure.py
portfolio\signals\trend.py
portfolio\signals\treasury_risk_rotation.py
portfolio\signals\structure.py
portfolio\signals\statistical_jump_regime.py
portfolio\signals\smart_money.py
portfolio\signals\shannon_entropy.py
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: conftest.py: The system cannot find the file specified. (os error 2)

 succeeded in 1030ms:
 M data/metals_swing_state.json
 M data/morning_briefing.json
 M docs/SESSION_PROGRESS.md
 M portfolio/loop_contract.py
?? .claude/commands/avanza-search.md
?? .claude/commands/digest-project.md
?? .claude/commands/fin-prereview.md
?? .claude/commands/time.md
?? .openclaw/
?? .playwright-cli/
?? .playwright-mcp/
?? .venv-unsloth/
?? 0
?? AGENTS.md
?? CLAUDE.local.md
?? HEARTBEAT.md
?? IDENTITY.md
?? SOUL.md
?? TOOLS.md
?? USER.md
?? data/_arxiv_gex.xml
?? data/_arxiv_gex2.xml
?? data/_arxiv_ivskew.xml
?? data/_arxiv_q1.xml
?? data/_arxiv_query1.xml
?? data/_arxiv_stable.xml
?? data/_arxiv_te.xml
?? data/_mini_s_report.txt
?? data/_mini_s_results.txt
?? data/_new_warrant_details.json
?? data/_signal_report_tmp.txt
?? data/_silver_warrants_report.txt
?? data/_ss_q1.json
?? data/_ss_q2.json
?? data/_ss_query1.json
?? data/accuracy_snapshot_2026-04-18.json
?? data/accuracy_snapshot_state.json
?? data/adversarial_review_out.txt
?? data/after-hours-research-log.jsonl
?? data/after-hours-research-out.txt
?? data/agent_context_t1.json
?? data/agent_context_t2.json
?? data/algo_trading_research_2026-04-08.json
?? data/archive/
?? data/arm_stop_orders_once.py
?? data/auto-improve-codex-log.jsonl
?? data/auto-improve-codex-model-probe.txt
?? data/auto-improve-codex-out.txt
?? data/auto-improve-codex-progress.json
?? data/auto-improve-log.jsonl
?? data/auto-improve-out.txt
?? data/autonomous_throttle.json
?? data/avanza_instruments_live.json
?? data/avanza_search_all.json
?? data/avanza_search_result.json
?? data/avanza_silver_warrants.json
?? data/backtest_conf_threshold/
?? data/backtest_new_signals_out.json
?? data/backtest_silver_exit_rule.py
?? data/backtest_sjm_results.json
?? data/bench_history.jsonl
?? data/calibration_analysis_2026-04-18.json
?? data/check_llm_health.py
?? data/check_processes.ps1
?? data/claude_fundamental_log.jsonl
?? data/claude_invocations.jsonl
?? data/contract_state.json
?? data/contract_violations.jsonl
?? data/cot_history.jsonl
?? data/cpu_bench.py
?? data/cpu_bench_report.py
?? data/cpu_check.ps1
?? data/cpu_monitor.ps1
?? data/crypto_scheduler_state.json
?? data/daily_digest_state.json
?? data/daily_research_next_batch.json
?? data/degradation_alert_state.json
?? data/digest_state.json
?? data/disabled_signal_rescue_2026-04-18.json
?? data/docx_extract.txt
?? data/elongir_log.jsonl
?? data/elongir_state.json
?? data/elongir_trades.jsonl
?? data/exchange_netflow_history.jsonl
?? data/extract_docx.py
?? data/extract_pdf_oil.py
?? data/extract_pdf_oil2.py
?? data/fear_greed_streak.json
?? data/fetch_warrant_prices.py
?? data/fgl-logs/
?? data/fin_command_lessons.json
?? data/fin_command_log.jsonl
?? data/fin_evolve_state.json
?? data/fin_fish_log.jsonl
?? data/fin_fish_monitor.jsonl
?? data/fin_snipe_fills.jsonl
?? data/fin_snipe_manager_log.jsonl
?? data/fin_snipe_manager_out.txt
?? data/fin_snipe_predictions.jsonl
?? data/fin_snipe_state.json
?? data/fish_engine_state.json
?? data/fish_heartbeat.txt
?? data/fish_monitor.py
?? data/fish_monitor_log.jsonl
?? data/fish_monitor_state.json
?? data/fish_precomputed.json
?? data/fish_trades.jsonl
?? data/fishing_context.json
?? data/fix_md.py
?? data/fundamentals_cache.json
?? data/fx_rate_cache.json
?? data/gold_analysis_temp.py
?? data/gold_btc_ratio_history.jsonl
?? data/gold_deep_context.json
?? data/gold_refresh_state.json
?? data/gold_target_calc.py
?? data/golddigger.singleton.lock
?? data/golddigger_contract_state.json
?? data/golddigger_log.jsonl
?? data/golddigger_out.txt
?? data/golddigger_state.json
?? data/golddigger_trades.jsonl
?? data/grid_fisher_decisions.jsonl
?? data/grid_fisher_probe_decisions.jsonl
?? data/grid_fisher_probe_state.json
?? data/grid_fisher_state.json
?? data/headlines_latest.json
?? data/hw_history/
?? data/hw_monitor.json
?? data/ic_analysis_2026-04-18.json
?? data/ic_cache.json
?? data/journal_outcomes.jsonl
?? data/kronos_training/
?? data/layer2_decision_outcomes.jsonl
?? data/layer2_decisions.jsonl
?? data/layer2_execute.py
?? data/layer2_send_now.py
?? data/layer2_send_tg.py
?? data/layer2_tg_send.py
?? data/layer2_tg_xau_sell.py
?? data/layer2_write_journal_and_tg.py
?? data/list_python_procs.ps1
?? data/llama_server.pid
?? data/llm_backfill_out.txt
?? data/llm_probability_log.jsonl
?? data/llm_probability_outcomes.jsonl
?? data/llm_probability_outcomes.jsonl.bak-20260428
?? data/llm_rotation_state.jsonl
?? data/local_llm_report_export_state.json
?? data/local_llm_report_history.jsonl
?? data/local_llm_report_latest.json
?? data/loop_stderr.txt
?? data/loop_stdout.txt
?? data/main_loop.singleton.lock
?? data/market_health_state.json
?? data/meta_learner_retrain_out.txt
?? data/metals_agent_summary.json
?? data/metals_contract_state.json
?? data/metals_decisions.jsonl
?? data/metals_ghost_positions.json
?? data/metals_guard_state.json
?? data/metals_llm_outcomes.jsonl
?? data/metals_llm_predictions.jsonl
?? data/metals_loop.singleton.lock
?? data/metals_loop_out.txt
?? data/metals_news_summary.json
?? data/metals_precompute_state.json
?? data/metals_refresh_state.json
?? data/metals_signal_accuracy.json
?? data/metals_signal_log.jsonl
?? data/metals_signal_outcomes.jsonl
?? data/metals_spike_state.json
?? data/metals_swing_decisions.jsonl
?? data/metals_trades.jsonl
?? data/metals_value_history.jsonl
?? data/metals_warrant_catalog.json
?? data/microstructure_state.json
?? data/models/
?? data/monitor_silver_exit.stdout
?? data/mrmr_analysis_2026-04-18.json
?? data/mstr_loop_backtest_results.json
?? data/mstr_loop_poll.jsonl
?? data/mstr_loop_shadow.jsonl
?? data/mstr_loop_state.json
?? data/mstr_loop_telegram_state.json
?? data/oil_cot_history.jsonl
?? data/oil_deep_context.json
?? data/oil_grid_signal.json
?? data/oil_pdf2_text.txt
?? data/oil_pdf_text.txt
?? data/oil_precompute_state.json
?? data/oil_refresh_state.json
?? data/onchain_cache.json
?? data/outcome_check_out.txt
?? data/per_ticker_ic_summary_2026-04-18.json
?? data/portfolio.log.1
?? data/portfolio.log.2
?? data/portfolio.log.3
?? data/price_snapshots_hourly.jsonl
?? data/quant_research_findings.json
?? data/rc-server-2_out.txt
?? data/rc-server-3_out.txt
?? data/rc-server_out.txt
?? data/read_temps.ps1
?? data/regime_accuracy_cache.json
?? data/regime_analysis_2026-04-18.json
?? data/research_literature_scan_2026-04-10.json
?? data/seasonality_profiles.json
?? data/semgrep_baseline.json
?? data/sentiment_shadow_outcomes.jsonl
?? data/shadow_review_out.txt
?? data/signal-research-log.jsonl
?? data/signal-research-progress.json
?? data/signal_candidates.db
?? data/signal_decay_alerts.jsonl
?? data/signal_postmortem.json
?? data/signal_research_2026-04-28.json
?? data/signal_research_backlog.db
?? data/signal_research_backlog.jsonl
?? data/signal_research_backtest.json
?? data/signal_research_out.txt
?? data/signal_research_papers.json
?? data/signal_research_ranked.json
?? data/signal_research_summary.json
?? data/signal_research_web.json
?? data/signal_state_since.json
?? data/signal_utility_cache.json
?? data/signal_weights.json
?? data/silver_analysis.json
?? data/silver_deep_context.json
?? data/silver_fomc_loop.py
?? data/silver_fomc_monitor.py
?? data/silver_monitor_err.txt
?? data/silver_monitor_out.txt
?? data/silver_refresh_state.json
?? data/silver_research.md
?? data/silver_research.md.tmp.44552.1773071148621
?? data/silver_tg_send.py
?? data/stack_overflow_counter.json
?? data/start_loops.bat
?? data/start_loops.py
?? data/system_lessons.json
?? data/tg_layer2_analysis.py
?? data/tg_layer2_eth.py
?? data/tg_layer2_mstr.py
?? data/tg_layer2_pltr.py
?? data/tg_layer2_quick.py
?? data/tg_layer2_send.py
?? data/tg_layer2_send_now.py
?? data/tg_layer2_t2.py
?? data/tg_layer2_xau.py
?? data/tg_send_analysis.py
?? data/tg_send_layer2.py
?? data/ticker_signal_accuracy_cache.json
?? data/trade_guard_state.json
?? data/tune_new_signals_out.json
?? data/weekly_research_2026-03-29.json
?? "docs/Configure Codex to reduce command permission prompts for my development\342\200\246.md"
?? docs/GOLDDIGGER_FINAL.md
?? docs/Oil-research-2.pdf
?? docs/Oil-research.pdf
?? docs/PLAN-trigger-noise.md
?? docs/PLUGINS_AND_SKILLS.md
?? docs/SYSTEM_HEALTH_CONTRACT.md
?? docs/UNSLOTH_RUNTIME_LEARNINGS_2026-03-17.md
?? docs/adversarial-review-2026-05-11/
?? docs/auto-improve-prompt-codex.md
?? "docs/codex guidelines.md"
?? "docs/deep research/"
?? docs/oil-deep-research-report-2.md
?? docs/oil-deep-research-report.md
?? docs/openclaw/
?? docs/plans/2026-03-17-merge-triage-plan.md
?? docs/plans/2026-03-18-unsloth-finetune-plan.md
?? docs/plans/2026-04-16-gemma4-loop-plan.md
?? docs/plans/2026-04-18-ic-weighting-integration.md
?? docs/reviews/
?? docs/superpowers/plans/2026-03-27-3h-signal-optimization.md
?? docs/superpowers/plans/2026-03-30-househunting.md
?? docs/superpowers/plans/2026-03-31-metals-microstructure-signals.md
?? docs/superpowers/plans/2026-04-04-strategy-orchestrator-merge.md
?? docs/superpowers/plans/2026-04-28-dashboard-ops-board.md
?? docs/superpowers/specs/2026-03-30-househunting-design.md
?? docs/superpowers/specs/2026-04-28-dashboard-ops-board-design.md
?? home_phone_full.png
?? memory/
?? phone-assets-loaded.png
?? phone-assets.png
?? phone-avanza-live.png
?? phone-avanza.png
?? phone-bottomsheet.png
?? phone-dark.png
?? phone-decision-detail.png
?? phone-decisions.png
?? phone-health.png
?? phone-home-final.png
?? phone-home.png
?? phone-more.png
?? phone-prices.png
?? phone-refresh-clicked.png
?? phone-refresh-flash.png
?? phone-settings.png
?? phone-signals.png
?? portfolio_phone.png
?? scripts/auto-improve-codex.bat
?? scripts/avanza_metals_check.py
?? scripts/avanza_metals_ladder.py
?? scripts/backtest_new_signals.py
?? scripts/benchmark_gpu_models.py
?? scripts/cleanup_settings_20260508.sh
?? scripts/fin_snipe.py
?? scripts/fin_snipe_manager.py
?? scripts/monitor_silver_exit.py
?? scripts/prepare_kronos_training_data.py
?? scripts/setup_wsl_claude.ps1
?? scripts/signal_correlation_audit.py
?? scripts/signal_research_extract.py
?? scripts/signal_research_phase34.py
?? scripts/sysmon.py
?? scripts/tune_new_signals.py
?? scripts/win/add-cloudflared-path.ps1
?? scripts/win/fin-snipe-manager.bat
?? scripts/win/fin-snipe.bat
?? scripts/win/golddigger-loop.bat
?? scripts/win/install-golddigger-task.ps1
?? scripts/win/install-health-check-tasks.ps1
?? scripts/win/install-loop-resume-task.ps1
?? scripts/win/install-market-tasks.ps1
?? scripts/win/install-metals-loop-task.ps1
?? scripts/win/install-rc-keepalive-task.ps1
?? scripts/win/install-rc-server-task.ps1
?? scripts/win/install-rc-watchdog-task.ps1
?? scripts/win/metals-arm-stop-once.bat
?? scripts/win/pf-loop-ensure.ps1
?? scripts/win/rc-keepalive.ps1
?? scripts/win/rc-watchdog.ps1
?? scripts/win/settings-cleanup-20260508.bat
?? scripts/win/silver-monitor.bat
?? scripts/write_research_outputs.py
?? training/unsloth/
?? unsloth_compiled_cache/
?? worktrees/
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
warning: could not open directory '.pytest_tmp/': Permission denied

codex
I’ve confirmed the repo is dirty, so I’m treating everything as read-only evidence and not inferring intent from uncommitted files. Next I’m walking the canonical I/O and locking layers first, because the rest of the dashboard and scripts depend on those invariants.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "''^" portfolio/file_utils.py' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "''^" portfolio/shared_state.py' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "''^" portfolio/process_lock.py' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "''^" portfolio/subprocess_utils.py' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "''^" portfolio/config_validator.py' in Q:\finance-analyzer
 succeeded in 755ms:
1:"""Shared file I/O utilities."""
2:import json
3:import logging
4:import os
5:import tempfile
6:from collections import deque
7:from contextlib import contextmanager, suppress
8:from pathlib import Path
9:
10:# Cross-platform file-locking primitives for `atomic_append_jsonl`.
11:# Same pattern as `portfolio/process_lock.py`.
12:try:
13:    import msvcrt as _msvcrt  # type: ignore[import-not-found]
14:except ImportError:  # pragma: no cover - non-Windows
15:    _msvcrt = None  # type: ignore[assignment]
16:try:
17:    import fcntl as _fcntl  # type: ignore[import-not-found]
18:except ImportError:  # pragma: no cover - Windows
19:    _fcntl = None  # type: ignore[assignment]
20:
21:logger = logging.getLogger("portfolio.file_utils")
22:
23:
24:def atomic_write_text(path, text, encoding="utf-8"):
25:    """Atomically write text to a file using tempfile + os.replace.
26:
27:    Same safety guarantees as atomic_write_json: fsync before replace,
28:    no partial writes on crash.
29:    """
30:    path = Path(path)
31:    path.parent.mkdir(parents=True, exist_ok=True)
32:    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
33:    try:
34:        with os.fdopen(fd, "w", encoding=encoding) as f:
35:            f.write(text)
36:            f.flush()
37:            os.fsync(f.fileno())
38:        os.replace(tmp, str(path))
39:    except BaseException:
40:        with suppress(OSError):
41:            os.unlink(tmp)
42:        raise
43:
44:
45:def atomic_write_json(path, data, indent=2, ensure_ascii=True):
46:    """Atomically write JSON data to a file using tempfile + os.replace.
47:
48:    Ensures the file is never left in a partially-written state.
49:    Fsyncs before replace to guarantee durability on power loss (H34).
50:    """
51:    path = Path(path)
52:    path.parent.mkdir(parents=True, exist_ok=True)
53:    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
54:    try:
55:        with os.fdopen(fd, "w", encoding="utf-8") as f:
56:            json.dump(data, f, indent=indent, default=str, ensure_ascii=ensure_ascii)
57:            f.flush()
58:            os.fsync(f.fileno())
59:        os.replace(tmp, str(path))
60:    except BaseException:
61:        with suppress(OSError):
62:            os.unlink(tmp)
63:        raise
64:
65:
66:def load_json(path, default=None):
67:    """Load a JSON file. Returns *default* if missing or unparseable.
68:
69:    Uses try/except instead of exists() check to avoid TOCTOU race.
70:    Handles OSError (permission denied, locked files) gracefully on Windows.
71:    Logs WARNING on corrupt JSON so corruption is observable (H35).
72:    """
73:    path = Path(path)
74:    try:
75:        return json.loads(path.read_text(encoding="utf-8"))
76:    except FileNotFoundError:
77:        return default
78:    except OSError:
79:        # BUG-139: PermissionError (file locked by antivirus/another process)
80:        # and other OS-level errors should degrade gracefully like missing files.
81:        logger.debug("load_json: OS error reading %s, returning default", path.name)
82:        return default
83:    except (json.JSONDecodeError, ValueError):
84:        # H35: Log corruption so it's observable — silent defaults hide data loss.
85:        logger.warning("load_json: corrupt JSON in %s, returning default", path.name)
86:        return default
87:
88:
89:def require_json(path):
90:    """Load a JSON file, raising on corruption or missing file.
91:
92:    Unlike load_json(), this function does NOT silently return defaults.
93:    Use for critical files where corruption must be surfaced (H35).
94:
95:    Raises:
96:        FileNotFoundError: If the file does not exist.
97:        json.JSONDecodeError: If the file contains invalid JSON.
98:        OSError: If the file cannot be read.
99:    """
100:    path = Path(path)
101:    return json.loads(path.read_text(encoding="utf-8"))
102:
103:
104:def load_jsonl(path, limit=None):
105:    """Load entries from a JSONL file.
106:
107:    Args:
108:        path: Path to the .jsonl file.
109:        limit: If set, keep only the *last* N entries (uses a deque).
110:
111:    Returns:
112:        list of parsed dicts. Empty list if file missing or empty.
113:    """
114:    path = Path(path)
115:    container = deque(maxlen=limit) if limit else []
116:    try:
117:        f = open(path, encoding="utf-8")
118:    except FileNotFoundError:
119:        return []
120:    except OSError as e:
121:        logger.warning("load_jsonl: cannot open %s: %s", path.name, e)
122:        return []
123:    with f:
124:        for line in f:
125:            line = line.strip()
126:            if not line:
127:                continue
128:            try:
129:                container.append(json.loads(line))
130:            except json.JSONDecodeError as e:
131:                logger.debug("Skipping malformed JSONL line in %s: %s", path.name, str(e)[:100])
132:                continue
133:    return list(container)
134:
135:
136:def load_jsonl_tail(path, max_entries=500, tail_bytes=512_000):
137:    """Load the last N entries from a JSONL file by reading from the end.
138:
139:    Much more efficient than load_jsonl(limit=N) for large files because
140:    it only reads the last `tail_bytes` bytes instead of the entire file.
141:
142:    Args:
143:        path: Path to the .jsonl file.
144:        max_entries: Maximum entries to return.
145:        tail_bytes: How many bytes to read from the end of the file.
146:            Default 512KB ≈ ~1000 typical entries.
147:
148:    Returns:
149:        list of parsed dicts (chronological order). Empty list if missing.
150:    """
151:    path = Path(path)
152:    try:
153:        file_size = path.stat().st_size
154:    except (FileNotFoundError, OSError):
155:        return []
156:    if file_size == 0:
157:        return []
158:
159:    entries = []
160:    try:
161:        with open(path, "rb") as f:
162:            # Seek to near end of file
163:            offset = max(0, file_size - tail_bytes)
164:            # 2026-05-04 codex P3-1 follow-up: peek the byte just before
165:            # the seek point. If it's a newline, the seek lands exactly
166:            # at a line boundary and the first decoded line is intact.
167:            # Without this check, a happy-coincidence boundary would
168:            # cost us one valid entry on every read.
169:            seek_on_boundary = False
170:            if offset > 0:
171:                f.seek(offset - 1)
172:                prior = f.read(1)
173:                seek_on_boundary = prior == b"\n"
174:            f.seek(offset)
175:            data = f.read()
176:        # Decode and split into lines
177:        text = data.decode("utf-8", errors="replace")
178:        lines = text.split("\n")
179:        # Drop the first line only when we landed mid-line. When seek
180:        # lands on a newline boundary, the first decoded line is
181:        # complete and should be kept.
182:        if offset > 0 and lines and not seek_on_boundary:
183:            lines = lines[1:]
184:        for line in lines:
185:            line = line.strip()
186:            if not line:
187:                continue
188:            try:
189:                entries.append(json.loads(line))
190:            except json.JSONDecodeError:
191:                continue
192:    except (OSError, UnicodeDecodeError) as e:
193:        logger.debug("load_jsonl_tail failed for %s: %s", path.name, e)
194:        return []
195:
196:    # Return last max_entries in chronological order
197:    if len(entries) > max_entries:
198:        entries = entries[-max_entries:]
199:    return entries
200:
201:
202:@contextmanager
203:def jsonl_sidecar_lock(path):
204:    """Yield while holding an exclusive sidecar lock keyed off *path*.
205:
206:    Same locking primitive that ``atomic_append_jsonl`` uses, exposed as
207:    a context manager so other code (notably
208:    ``portfolio.log_rotation.rotate_jsonl``) can serialize against
209:    in-flight appends. Lock file is ``<path.parent>/.<path.name>.lock``;
210:    a single-byte range is locked exclusively.
211:
212:    Pattern rationale:
213:
214:    * **Sidecar (not target):** locking the target file itself is racy
215:      when it is brand-new / size 0 — two first-writers can both hit
216:      the empty-file ``msvcrt.locking(fd, LK_LOCK, 1)`` failure path
217:      and interleave. A pre-seeded sidecar guarantees a lockable byte
218:      always exists.
219:    * **Windows + POSIX:** ``msvcrt.locking`` blocks on contention on
220:      Windows; ``fcntl.flock`` blocks on POSIX. Both release on close.
221:
222:    Callers MUST keep *all* mutations of the target file inside the
223:    ``with`` block — read, write, fsync, rename. Appends that arrive
224:    between rotation's "read all lines" and ``os.replace`` would
225:    otherwise be silently discarded (the divergence behind the
226:    ``signal_log_reconciliation`` contract invariant escalations of
227:    2026-05-11).
228:    """
229:    path = Path(path)
230:    path.parent.mkdir(parents=True, exist_ok=True)
231:    lock_path = path.parent / f".{path.name}.lock"
232:    if not lock_path.exists():
233:        try:
234:            with open(lock_path, "ab") as lf:
235:                if lf.tell() == 0:
236:                    lf.write(b"\0")
237:        except OSError:
238:            pass  # best-effort; lock open below will retry
239:
240:    with open(lock_path, "rb+") as lock_f:
241:        lfd = lock_f.fileno()
242:        win_locked = False
243:        try:
244:            if _msvcrt is not None:
245:                os.lseek(lfd, 0, os.SEEK_SET)
246:                _msvcrt.locking(lfd, _msvcrt.LK_LOCK, 1)  # blocking
247:                win_locked = True
248:            elif _fcntl is not None:
249:                _fcntl.flock(lfd, _fcntl.LOCK_EX)
250:            yield
251:        finally:
252:            if win_locked and _msvcrt is not None:
253:                try:
254:                    os.lseek(lfd, 0, os.SEEK_SET)
255:                    _msvcrt.locking(lfd, _msvcrt.LK_UNLCK, 1)
256:                except OSError:
257:                    pass
258:            # fcntl.flock releases automatically on close.
259:
260:
261:def atomic_append_jsonl(path, entry):
262:    """Append a single JSON entry to a JSONL file with atomic semantics
263:    across threads and processes.
264:
265:    Now built on :func:`jsonl_sidecar_lock` so the lock contract is
266:    shared with ``log_rotation.rotate_jsonl``. Without that contract,
267:    rotation's read → write-tmp → ``os.replace`` could discard any
268:    append that landed between read and replace — exactly the
269:    divergence the ``signal_log_reconciliation`` contract invariant
270:    detects.
271:
272:    This primitive is used by ~20 JSONL writers across the codebase
273:    (signal_log, claude_invocations, critical_errors, telegram_messages,
274:    accuracy_snapshots, etc.) so the fix eliminates torn-line risk
275:    system-wide. Unxfails
276:    ``tests/test_fix_agent_dispatcher.py::test_concurrent_append_does_not_corrupt_jsonl``.
277:    """
278:    path = Path(path)
279:    data = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
280:    with jsonl_sidecar_lock(path):
281:        with open(path, "ab") as f:
282:            f.write(data)
283:            f.flush()
284:            os.fsync(f.fileno())
285:
286:
287:def atomic_write_jsonl(path, entries):
288:    """Atomically rewrite a JSONL file with the given entries.
289:
290:    Uses tempfile + os.replace so the file is never left partially written.
291:    """
292:    path = Path(path)
293:    path.parent.mkdir(parents=True, exist_ok=True)
294:    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
295:    try:
296:        with os.fdopen(fd, "w", encoding="utf-8") as f:
297:            for entry in entries:
298:                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
299:            f.flush()
300:            os.fsync(f.fileno())
301:        os.replace(tmp, str(path))
302:    except BaseException:
303:        with suppress(OSError):
304:            os.unlink(tmp)
305:        raise
306:
307:
308:def last_jsonl_entry(path, field=None):
309:    """Return the last parsed JSON entry from a JSONL file (efficient tail read).
310:
311:    Reads only the last 4KB of the file instead of scanning the entire file.
312:
313:    Args:
314:        path: Path to the JSONL file.
315:        field: If set, return only this field's value from the last entry.
316:
317:    Returns:
318:        The last entry (dict) or the value of *field*, or None if file is
319:        missing/empty/unreadable.
320:    """
321:    path = Path(path)
322:    try:
323:        file_size = path.stat().st_size
324:    except (OSError, FileNotFoundError):
325:        return None
326:    if file_size == 0:
327:        return None
328:    read_size = min(file_size, 4096)
329:    try:
330:        with open(path, "rb") as f:
331:            f.seek(max(0, file_size - read_size))
332:            tail = f.read().decode("utf-8", errors="replace")
333:    except OSError:
334:        return None
335:    for line in reversed(tail.strip().splitlines()):
336:        line = line.strip()
337:        if not line:
338:            continue
339:        try:
340:            entry = json.loads(line)
341:            if field is not None:
342:                return entry.get(field)
343:            return entry
344:        except (json.JSONDecodeError, AttributeError):
345:            continue
346:    return None
347:
348:
349:def prune_jsonl(path, max_entries=5000):
350:    """Prune a JSONL file to keep only the most recent *max_entries*.
351:
352:    Reads the file, keeps the tail, and atomically rewrites it.
353:    Skips malformed lines (e.g., from partial writes) during read.
354:    No-op if the file has fewer entries than *max_entries*.
355:
356:    Returns the number of entries removed, or 0 if no pruning was needed.
357:    """
358:    path = Path(path)
359:    lines = []
360:    try:
361:        f = open(path, encoding="utf-8")
362:    except FileNotFoundError:
363:        return 0
364:    with f:
365:        for line in f:
366:            stripped = line.strip()
367:            if not stripped:
368:                continue
369:            # Validate JSON to avoid preserving corrupt partial lines
370:            try:
371:                json.loads(stripped)
372:                lines.append(stripped)
373:            except json.JSONDecodeError:
374:                logger.warning("prune_jsonl: skipping malformed line in %s", path.name)
375:    if len(lines) <= max_entries:
376:        return 0
377:    removed = len(lines) - max_entries
378:    keep = lines[-max_entries:]
379:    # Atomic rewrite via tempfile
380:    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
381:    try:
382:        with os.fdopen(fd, "w", encoding="utf-8") as f:
383:            for line in keep:
384:                f.write(line + "\n")
385:            f.flush()
386:            os.fsync(f.fileno())
387:        os.replace(tmp, str(path))
388:    except BaseException:
389:        with suppress(OSError):
390:            os.unlink(tmp)
391:        raise
392:    logger.info("Pruned %s: removed %d entries, kept %d", path.name, removed, max_entries)
393:    return removed
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 753ms:
1:"""Shared mutable state for the portfolio system.
2:
3:All modules that need caching or cross-cycle state import from here.
4:This avoids circular imports and provides a single source of truth.
5:"""
6:
7:import logging
8:import threading
9:import time
10:from datetime import UTC
11:
12:logger = logging.getLogger("portfolio.shared_state")
13:
14:# Tool cache — avoid re-running expensive tools every cycle
15:_tool_cache = {}
16:
17:# Retry cooldown for _cached() errors
18:_RETRY_COOLDOWN = 60
19:
20:
21:_CACHE_MAX_SIZE = 512  # evict expired entries when cache exceeds this size
22:_cache_lock = threading.Lock()
23:
24:# BUG-166: Dogpile/thundering-herd prevention.
25:# Tracks which keys are currently being refreshed. When a thread sees a cache
26:# miss and the key is already loading, it returns stale data (if available)
27:# instead of calling the function redundantly.
28:_loading_keys: set[str] = set()
29:_LOADING_TIMEOUT = 120  # seconds to wait for a loading thread before giving up
30:# C11/SS1: Track when each key was added to _loading_keys for eviction of
31:# permanently stuck keys (batch flush crash before updating cache).
32:_loading_timestamps: dict[str, float] = {}
33:
34:_MAX_STALE_FACTOR = 3  # return None if cached data is older than TTL * this factor
35:
36:
37:def _cached(key, ttl, func, *args):
38:    """Cache-through helper: returns cached data if fresh, else calls func.
39:
40:    Dogpile prevention (BUG-166): when multiple threads detect a cache miss
41:    simultaneously, only one thread fetches the data. Others return stale
42:    data if available, preventing redundant expensive calls (LLM inference,
43:    API requests) and model swap contention.
44:
45:    On error, returns stale data if it's less than TTL * _MAX_STALE_FACTOR old.
46:    Beyond that, returns None to prevent trading on dangerously old data.
47:    """
48:    now = time.time()
49:    with _cache_lock:
50:        if key in _tool_cache and now - _tool_cache[key]["time"] < ttl:
51:            return _tool_cache[key]["data"]
52:        # Evict expired entries when cache grows too large
53:        # Use TTL-aware eviction: entries expire after ttl * _MAX_STALE_FACTOR
54:        if len(_tool_cache) > _CACHE_MAX_SIZE:
55:            expired = [k for k, v in _tool_cache.items()
56:                       if now - v["time"] > v.get("ttl", 3600) * _MAX_STALE_FACTOR]
57:            for k in expired:
58:                del _tool_cache[k]
59:            # LRU fallback: if still over limit (all entries fresh), evict oldest 25%
60:            if len(_tool_cache) > _CACHE_MAX_SIZE:
61:                sorted_keys = sorted(
62:                    _tool_cache, key=lambda k: _tool_cache[k]["time"]
63:                )
64:                evict_count = len(sorted_keys) // 4 or 1
65:                for k in sorted_keys[:evict_count]:
66:                    del _tool_cache[k]
67:
68:        # C11/SS1: Evict stuck loading keys older than _LOADING_TIMEOUT seconds.
69:        _now_evict = time.time()
70:        stuck = [k for k, ts in _loading_timestamps.items()
71:                 if _now_evict - ts > _LOADING_TIMEOUT]
72:        for k in stuck:
73:            _loading_keys.discard(k)
74:            stuck_duration = _now_evict - _loading_timestamps.pop(k, _now_evict)
75:            logger.warning("[%s] evicted stuck loading key after %.0fs (timeout %ds)", k, stuck_duration, _LOADING_TIMEOUT)
76:
77:        # BUG-166: Dogpile prevention — if another thread is already loading
78:        # this key, return stale data instead of calling func redundantly.
79:        if key in _loading_keys:
80:            if key in _tool_cache:
81:                age = now - _tool_cache[key]["time"]
82:                max_stale = ttl * _MAX_STALE_FACTOR
83:                if age <= max_stale:
84:                    logger.debug("[%s] stale-while-revalidate (another thread loading)", key)
85:                    return _tool_cache[key]["data"]
86:            # No stale data available — return None rather than pile on
87:            logger.debug("[%s] no stale data, another thread loading — returning None", key)
88:            return None
89:        _loading_keys.add(key)
90:        _loading_timestamps[key] = time.time()
91:
92:    try:
93:        data = func(*args)
94:        with _cache_lock:
95:            # Don't cache None results — they typically indicate transient
96:            # API failures that returned None instead of raising. Caching
97:            # None hides the failure for the entire TTL duration, preventing
98:            # retry. Stale data (if any) is preserved for fallback.
99:            if data is not None:
100:                _tool_cache[key] = {"data": data, "time": now, "ttl": ttl}
101:            _loading_keys.discard(key)
102:            _loading_timestamps.pop(key, None)  # BUG-213: clean up on success path
103:        return data
104:    except KeyboardInterrupt:
105:        with _cache_lock:
106:            _loading_keys.discard(key)
107:            _loading_timestamps.pop(key, None)
108:        logger.warning("[%s] interrupted (KeyboardInterrupt), returning None", key)
109:        return None
110:    except Exception as e:
111:        logger.warning("[%s] error: %s", key, e)
112:        with _cache_lock:
113:            _loading_keys.discard(key)
114:            _loading_timestamps.pop(key, None)
115:            if key in _tool_cache:
116:                age = now - _tool_cache[key]["time"]
117:                max_stale = ttl * _MAX_STALE_FACTOR
118:                if age > max_stale:
119:                    logger.warning(
120:                        "[%s] stale data too old (%.0fs > %.0fs max), returning None",
121:                        key, age, max_stale,
122:                    )
123:                    return None
124:                _tool_cache[key]["time"] = now - ttl + _RETRY_COOLDOWN
125:                return _tool_cache[key]["data"]
126:        return None
127:
128:
129:def _cached_or_enqueue(key, ttl, enqueue_fn, context,
130:                        should_enqueue_fn=None, max_stale_factor=None):
131:    """Check cache — if fresh return it, if expired enqueue for batch and return stale.
132:
133:    Unlike _cached(), this never calls the model directly. On miss, it adds
134:    the request to the batch queue and returns stale data (or None).
135:
136:    Dogpile prevention (Codex finding #5): uses _loading_keys to avoid
137:    re-enqueuing the same key every cycle if the batch flush hasn't run yet.
138:
139:    2026-04-10 (perf/llama-swap-reduction) — two new optional parameters to
140:    support rotation scheduling of LLM signals:
141:
142:    - should_enqueue_fn: callable returning bool. If provided and the cache
143:      is stale-but-present, skip the enqueue when the callback says "no"
144:      (rotation off-cycle). If stale data is NOT available, force-enqueue
145:      regardless of the callback — we cannot leave the caller empty-handed
146:      when no stale fallback exists. Default None means "always enqueue",
147:      which preserves the pre-rotation behavior for every existing caller.
148:
149:    - max_stale_factor: integer override for how stale data can be returned,
150:      in multiples of ttl. Default None means use the module-level
151:      _MAX_STALE_FACTOR. LLM rotation passes 5 here so each rotated vote
152:      can stay valid across the full rotation cycle (3 * TTL) plus slippage.
153:    """
154:    now = time.time()
155:    effective_stale_factor = (
156:        max_stale_factor if max_stale_factor is not None else _MAX_STALE_FACTOR
157:    )
158:    with _cache_lock:
159:        if key in _tool_cache and now - _tool_cache[key]["time"] < ttl:
160:            return _tool_cache[key]["data"]
161:
162:        # Check stale availability BEFORE deciding whether to enqueue, because
163:        # the rotation gate can only safely skip enqueue when we have stale
164:        # fallback to return. If stale is exhausted we must force-enqueue.
165:        #
166:        # 2026-04-10 code-review finding N1: also treat cached `data is None`
167:        # as "stale NOT available". main.py writes _update_cache(key, None)
168:        # when a flush fails as a short-lived retry cooldown; without this
169:        # check, the rotation gate would skip enqueue on those None entries
170:        # and the caller would see None for up to 3 rotation cycles
171:        # (~3 minutes) before retry. The extra `is not None` guard force-
172:        # enqueues on failed-cache entries matching legacy recovery behavior.
173:        stale_data = None
174:        stale_available = False
175:        if key in _tool_cache:
176:            age = now - _tool_cache[key]["time"]
177:            cached_data = _tool_cache[key]["data"]
178:            if age <= ttl * effective_stale_factor and cached_data is not None:
179:                stale_available = True
180:                stale_data = cached_data
181:
182:        # Decide whether to enqueue:
183:        # - Default (no should_enqueue_fn): always enqueue (legacy behavior)
184:        # - Callback returns True: enqueue (rotation on-cycle, or force path)
185:        # - Callback returns False AND stale available: skip (rotation off-cycle,
186:        #   stale fallback carries us until next on-cycle)
187:        # - Callback returns False AND stale NOT available: enqueue anyway
188:        #   (fresh cold path; caller has no fallback, we must refresh)
189:        if should_enqueue_fn is None:
190:            should_enq = True
191:        else:
192:            try:
193:                should_enq = bool(should_enqueue_fn()) or not stale_available
194:            except Exception as e:
195:                logger.warning(
196:                    "[%s] should_enqueue_fn raised, defaulting to enqueue: %s",
197:                    key, e,
198:                )
199:                should_enq = True
200:
201:        if should_enq and enqueue_fn and context is not None and key not in _loading_keys:
202:            _loading_keys.add(key)
203:            # C11/SS1: Track enqueue time for stuck-key eviction.
204:            _loading_timestamps[key] = time.time()
205:            try:
206:                enqueue_fn(key, context)
207:            except Exception as e:
208:                _loading_keys.discard(key)
209:                _loading_timestamps.pop(key, None)
210:                logger.warning("[%s] enqueue_fn raised, key released: %s", key, e)
211:
212:        # Return stale if available
213:        if stale_available:
214:            return stale_data
215:    return None
216:
217:
218:# 2026-04-10 (perf/llama-swap-reduction): monotonic counter of full-LLM
219:# batch flushes that actually processed work. Drives rotation scheduling in
220:# portfolio.llm_batch.is_llm_on_cycle — incremented at the end of
221:# flush_llm_batch() iff at least one phase had queued items. In-memory only,
222:# resets to 0 on process start; on restart the rotation deterministically
223:# restarts at ministral with a cold-start warmup cycle that runs all LLMs.
224:_full_llm_cycle_count = 0
225:
226:
227:def _update_cache(key, data, ttl=None):
228:    """Update a cache entry directly (for batch flush results)."""
229:    with _cache_lock:
230:        _loading_keys.discard(key)
231:        # C11/SS1: Clean up timestamp when key is resolved.
232:        _loading_timestamps.pop(key, None)
233:        _tool_cache[key] = {
234:            "data": data,
235:            "time": time.time(),
236:            "ttl": ttl or 900,
237:        }
238:
239:
240:# Cycle counter — incremented at the start of each run() to invalidate per-cycle caches
241:_run_cycle_id = 0
242:
243:# Current market state — updated each run() cycle, used by data_collector for yfinance fallback
244:_current_market_state = "open"
245:
246:# Regime detection cache (invalidated each cycle)
247:# BUG-169: Protected by _regime_lock — accessed from 8 concurrent ThreadPoolExecutor threads
248:_regime_cache = {}
249:_regime_cache_cycle = 0
250:_regime_lock = threading.Lock()
251:
252:
253:# --- Rate limiters ---
254:
255:class _RateLimiter:
256:    """Token-bucket rate limiter. Sleeps when calls exceed rate."""
257:    def __init__(self, max_per_minute, name=""):
258:        self.interval = 60.0 / max_per_minute
259:        self.last_call = 0.0
260:        self.name = name
261:        self._lock = threading.Lock()
262:
263:    def wait(self):
264:        # BUG-212: Sleep OUTSIDE the lock to avoid blocking all 8 worker
265:        # threads. Calculate sleep duration under the lock, release it,
266:        # then sleep.
267:        # Fix: Reserve the next slot (last_call = last_call + interval)
268:        # BEFORE releasing the lock, so parallel threads see the reserved
269:        # time and calculate a longer wait instead of stampeding.
270:        wait_time = 0.0
271:        with self._lock:
272:            now = time.time()
273:            elapsed = now - self.last_call
274:            if elapsed < self.interval:
275:                wait_time = self.interval - elapsed
276:            # Reserve the next slot atomically — even if we haven't slept yet,
277:            # the next thread to enter will see this and wait longer.
278:            self.last_call = self.last_call + self.interval if wait_time > 0 else now
279:        if wait_time > 0:
280:            time.sleep(wait_time)
281:
282:
283:# H11/DC-R3-4: yfinance is not thread-safe. This lock is shared across all
284:# modules (fear_greed, golddigger/data_provider, data_collector) so that
285:# concurrent calls from the 8-worker ThreadPoolExecutor are serialized.
286:# data_collector.py imports this lock instead of defining its own.
287:yfinance_lock = threading.Lock()
288:
289:# Alpaca IEX: 200 req/min → target 150/min to leave headroom
290:_alpaca_limiter = _RateLimiter(150, "alpaca")
291:# Binance: 1200 weight/min → very generous, but space out slightly
292:_binance_limiter = _RateLimiter(600, "binance")
293:# Yahoo Finance (yfinance): no official limit, but be polite — 30/min
294:_yfinance_limiter = _RateLimiter(30, "yfinance")
295:
296:
297:# Alpha Vantage: 5 req/min free tier
298:_alpha_vantage_limiter = _RateLimiter(5, "alpha_vantage")
299:
300:
301:# NewsAPI: 100 req/day free tier — tiered priority system
302:# Budget: metals (XAU, XAG) get 20-min refresh during active hours (~84/day)
303:# All other tickers: Yahoo-only (0 NewsAPI calls)
304:# BTC/ETH: already served by CryptoCompare, not NewsAPI
305:_newsapi_daily_count = 0
306:_newsapi_daily_reset = 0.0  # timestamp of last reset
307:_NEWSAPI_DAILY_BUDGET = 90  # leave 10-call margin
308:_newsapi_lock = threading.Lock()
309:
310:# Tier 1 = 20-min TTL during active hours; Tier 2 = 3h; rest = Yahoo-only
311:_NEWSAPI_PRIORITY = {"XAU": 1, "XAG": 1, "MSTR": 2}
312:
313:# Better search queries — raw ticker symbols return sparse results on NewsAPI
314:_NEWSAPI_SEARCH_QUERIES = {
315:    "XAU": "gold AND (price OR market OR ounce OR bullion OR futures OR commodity)",
316:    "XAG": "silver AND (price OR market OR ounce OR bullion OR futures OR commodity)",
317:    "MSTR": "MicroStrategy OR MSTR",
318:}
319:
320:# Active monitoring: 08:00-22:00 CET = 07:00-21:00 UTC
321:_NEWSAPI_ACTIVE_START_UTC = 7
322:_NEWSAPI_ACTIVE_END_UTC = 21
323:
324:
325:def newsapi_quota_ok() -> bool:
326:    """Check if we still have NewsAPI quota today. Thread-safe."""
327:    global _newsapi_daily_count, _newsapi_daily_reset
328:    now = time.time()
329:    with _newsapi_lock:
330:        # Reset counter at midnight UTC
331:        from datetime import datetime
332:        today_start = datetime.now(UTC).replace(
333:            hour=0, minute=0, second=0, microsecond=0
334:        ).timestamp()
335:        if _newsapi_daily_reset < today_start:
336:            _newsapi_daily_count = 0
337:            _newsapi_daily_reset = now
338:        return _newsapi_daily_count < _NEWSAPI_DAILY_BUDGET
339:
340:
341:def newsapi_track_call():
342:    """Increment NewsAPI daily counter. Call after each successful API request."""
343:    global _newsapi_daily_count
344:    with _newsapi_lock:
345:        _newsapi_daily_count += 1
346:        if _newsapi_daily_count == _NEWSAPI_DAILY_BUDGET:
347:            logger.warning("NewsAPI daily budget exhausted (%d/%d), falling back to Yahoo",
348:                          _newsapi_daily_count, _NEWSAPI_DAILY_BUDGET)
349:
350:
351:def newsapi_ttl_for_ticker(ticker: str):
352:    """Dynamic TTL based on ticker priority and time of day.
353:
354:    Returns TTL in seconds, or None to skip NewsAPI for this ticker.
355:    Tier 1 (metals): 20-min during active hours (08:00-22:00 CET).
356:    Other tickers: None (Yahoo-only, saves budget for metals).
357:    """
358:    short = ticker.upper().replace("-USD", "")
359:    priority = _NEWSAPI_PRIORITY.get(short)
360:    if priority is None:
361:        return None
362:
363:    from datetime import datetime
364:    hour_utc = datetime.now(UTC).hour
365:    is_active = _NEWSAPI_ACTIVE_START_UTC <= hour_utc < _NEWSAPI_ACTIVE_END_UTC
366:
367:    if is_active:
368:        if priority == 1:
369:            return 1200   # 20 min — metals
370:        return 10800      # 3h — secondary (MSTR etc.)
371:    return None  # off-hours: Yahoo-only
372:
373:
374:def newsapi_search_query(ticker: str) -> str:
375:    """Optimized search query for NewsAPI. Falls back to ticker symbol."""
376:    short = ticker.upper().replace("-USD", "")
377:    return _NEWSAPI_SEARCH_QUERIES.get(short, short)
378:
379:
380:# TTL constants for tool caching
381:FUNDAMENTALS_TTL = 86400  # 24 hours
382:ONCHAIN_TTL = 43200      # 12 hours (on-chain data updates slowly)
383:FEAR_GREED_TTL = 300     # 5 min
384:SENTIMENT_TTL = 900      # 15 min
385:MINISTRAL_TTL = 900      # 15 min
386:ML_SIGNAL_TTL = 900      # 15 min
387:FUNDING_RATE_TTL = 900   # 15 min
388:VOLUME_TTL = 300         # 5 min
389:NEWSAPI_TTL = 1800       # 30 min fallback — overridden by newsapi_ttl_for_ticker()
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 756ms:
1:"""Cross-platform helpers for non-blocking singleton process locks."""
2:
3:from __future__ import annotations
4:
5:import contextlib
6:import os
7:from datetime import UTC, datetime
8:from pathlib import Path
9:from typing import IO
10:
11:try:
12:    import msvcrt  # type: ignore[attr-defined]
13:except ImportError:  # pragma: no cover - non-Windows
14:    msvcrt = None
15:
16:try:
17:    import fcntl  # type: ignore[import-not-found]
18:except ImportError:  # pragma: no cover - Windows
19:    fcntl = None
20:
21:
22:def acquire_lock_file(
23:    lock_path: str | Path,
24:    *,
25:    owner: str = "",
26:    metadata: dict | None = None,
27:) -> IO[str] | None:
28:    """Acquire a non-blocking file lock and return the open handle.
29:
30:    Returns None if another process already holds the lock.
31:    """
32:    path = Path(lock_path)
33:    if path.parent:
34:        path.parent.mkdir(parents=True, exist_ok=True)
35:
36:    fh = path.open("a+", encoding="utf-8")
37:    try:
38:        _lock_file(fh)
39:    except OSError:
40:        fh.close()
41:        return None
42:
43:    _write_lock_metadata(fh, owner=owner, metadata=metadata)
44:    return fh
45:
46:
47:def release_lock_file(fh: IO[str] | None) -> None:
48:    """Release a previously acquired lock handle."""
49:    if fh is None:
50:        return
51:    try:
52:        _unlock_file(fh)
53:    except OSError:
54:        pass
55:    finally:
56:        with contextlib.suppress(Exception):
57:            fh.close()
58:
59:
60:def _lock_file(fh: IO[str]) -> None:
61:    fh.seek(0)
62:    if msvcrt is not None:
63:        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
64:        return
65:    if fcntl is not None:  # pragma: no branch - platform-specific
66:        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
67:        return
68:    raise RuntimeError(
69:        "No file locking mechanism available (neither msvcrt nor fcntl). "
70:        "Cannot guarantee mutual exclusion."
71:    )
72:
73:
74:def _unlock_file(fh: IO[str]) -> None:
75:    fh.seek(0)
76:    if msvcrt is not None:
77:        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
78:        return
79:    if fcntl is not None:  # pragma: no branch - platform-specific
80:        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
81:
82:
83:def _write_lock_metadata(
84:    fh: IO[str],
85:    *,
86:    owner: str = "",
87:    metadata: dict | None = None,
88:) -> None:
89:    payload = {
90:        "pid": os.getpid(),
91:        "started": datetime.now(UTC).isoformat(),
92:    }
93:    if owner:
94:        payload["owner"] = owner
95:    if metadata:
96:        payload.update({str(k): v for k, v in metadata.items() if v is not None})
97:
98:    try:
99:        fh.seek(0)
100:        fh.truncate()
101:        fh.write(" ".join(f"{key}={value}" for key, value in payload.items()) + "\n")
102:        fh.flush()
103:    except Exception:
104:        pass
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 784ms:
1:"""Subprocess utilities to prevent orphaned child processes on Windows.
2:
3:Provides:
4:- run_safe(): Drop-in subprocess.run() replacement that uses Windows Job Objects
5:  with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE so children die when the parent exits.
6:- popen_in_job(): Popen wrapper for long-running subprocesses — assigns the child
7:  to a Job Object so it's automatically killed if the parent dies.
8:- kill_orphaned_by_cmdline(): Find and kill orphaned processes matching a command
9:  line pattern (safety net for processes that escaped Job Object protection).
10:- kill_orphaned_llama(): Safety-net reaper for orphaned llama-completion.exe processes.
11:"""
12:
13:import json
14:import logging
15:import subprocess
16:import sys
17:
18:logger = logging.getLogger("portfolio.subprocess_utils")
19:
20:
21:def run_safe(cmd, **kwargs):
22:    """Run a subprocess with Windows Job Object protection.
23:
24:    Drop-in replacement for subprocess.run().  On Windows, creates a Job Object
25:    with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE so that the child process is
26:    automatically killed if the parent Python process dies.
27:
28:    Falls back to plain subprocess.run() on non-Windows or if Job Object
29:    creation fails.
30:
31:    Supported kwargs: capture_output, text, timeout, input, stdin (and any
32:    others accepted by subprocess.Popen / subprocess.run).
33:    """
34:    if sys.platform != "win32":
35:        return subprocess.run(cmd, **kwargs)
36:
37:    try:
38:        return _run_with_job_object(cmd, **kwargs)
39:    except Exception as exc:
40:        logger.debug("Job Object creation failed (%s), falling back to subprocess.run", exc)
41:        return subprocess.run(cmd, **kwargs)
42:
43:
44:def _create_job_object():
45:    """Create a Windows Job Object with KILL_ON_JOB_CLOSE.
46:
47:    Returns (job_handle, kernel32) or raises OSError.
48:    """
49:    import ctypes
50:    from ctypes import wintypes
51:
52:    kernel32 = ctypes.windll.kernel32
53:
54:    job = kernel32.CreateJobObjectW(None, None)
55:    if not job:
56:        raise OSError("CreateJobObjectW failed")
57:
58:    # JOBOBJECT_BASIC_LIMIT_INFORMATION (64-bit layout)
59:    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
60:        _fields_ = [
61:            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
62:            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
63:            ("LimitFlags", wintypes.DWORD),
64:            ("_pad0", wintypes.DWORD),
65:            ("MinimumWorkingSetSize", ctypes.c_size_t),
66:            ("MaximumWorkingSetSize", ctypes.c_size_t),
67:            ("ActiveProcessLimit", wintypes.DWORD),
68:            ("_pad1", wintypes.DWORD),
69:            ("Affinity", ctypes.c_size_t),
70:            ("PriorityClass", wintypes.DWORD),
71:            ("SchedulingClass", wintypes.DWORD),
72:        ]
73:
74:    class IO_COUNTERS(ctypes.Structure):
75:        _fields_ = [
76:            ("ReadOperationCount", ctypes.c_ulonglong),
77:            ("WriteOperationCount", ctypes.c_ulonglong),
78:            ("OtherOperationCount", ctypes.c_ulonglong),
79:            ("ReadTransferCount", ctypes.c_ulonglong),
80:            ("WriteTransferCount", ctypes.c_ulonglong),
81:            ("OtherTransferCount", ctypes.c_ulonglong),
82:        ]
83:
84:    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
85:        _fields_ = [
86:            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
87:            ("IoInfo", IO_COUNTERS),
88:            ("ProcessMemoryLimit", ctypes.c_size_t),
89:            ("JobMemoryLimit", ctypes.c_size_t),
90:            ("PeakProcessMemoryUsed", ctypes.c_size_t),
91:            ("PeakJobMemoryUsed", ctypes.c_size_t),
92:        ]
93:
94:    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
95:    JobObjectExtendedLimitInformation = 9
96:
97:    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
98:    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
99:
100:    ok = kernel32.SetInformationJobObject(
101:        job,
102:        JobObjectExtendedLimitInformation,
103:        ctypes.byref(info),
104:        ctypes.sizeof(info),
105:    )
106:    if not ok:
107:        kernel32.CloseHandle(job)
108:        raise OSError("SetInformationJobObject failed")
109:
110:    return job, kernel32
111:
112:
113:def _run_with_job_object(cmd, **kwargs):
114:    """Internal: run subprocess inside a Windows Job Object."""
115:    job, kernel32 = _create_job_object()
116:
117:    try:
118:        popen_kwargs = dict(kwargs)
119:        timeout = popen_kwargs.pop("timeout", None)
120:
121:        if popen_kwargs.pop("capture_output", False):
122:            popen_kwargs["stdout"] = subprocess.PIPE
123:            popen_kwargs["stderr"] = subprocess.PIPE
124:
125:        input_data = popen_kwargs.pop("input", None)
126:        if input_data is not None and "stdin" not in popen_kwargs:
127:            popen_kwargs["stdin"] = subprocess.PIPE
128:
129:        proc = subprocess.Popen(cmd, **popen_kwargs)
130:
131:        try:
132:            kernel32.AssignProcessToJobObject(job, int(proc._handle))
133:        except Exception as e:
134:            logger.warning(
135:                "Job Object assignment failed for pid %d — child may orphan: %s",
136:                proc.pid, e,
137:            )
138:
139:        try:
140:            stdout, stderr = proc.communicate(input=input_data, timeout=timeout)
141:        except subprocess.TimeoutExpired:
142:            proc.kill()
143:            proc.communicate()
144:            raise
145:
146:        return subprocess.CompletedProcess(
147:            args=cmd,
148:            returncode=proc.returncode,
149:            stdout=stdout,
150:            stderr=stderr,
151:        )
152:    finally:
153:        kernel32.CloseHandle(job)
154:
155:
156:def popen_in_job(cmd, **kwargs):
157:    """Start a long-running subprocess inside a Windows Job Object.
158:
159:    Like subprocess.Popen(), but assigns the child to a Job Object with
160:    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. If the parent process dies (crash,
161:    kill -9, power loss), the OS automatically kills the child.
162:
163:    Returns (proc, job_handle) on Windows. On non-Windows or if Job Object
164:    creation fails, returns (proc, None).
165:
166:    Caller must call close_job(job_handle) when explicitly stopping the child.
167:    """
168:    proc = subprocess.Popen(cmd, **kwargs)
169:
170:    if sys.platform != "win32":
171:        return proc, None
172:
173:    try:
174:        job, kernel32 = _create_job_object()
175:        kernel32.AssignProcessToJobObject(job, int(proc._handle))
176:        return proc, job
177:    except Exception as exc:
178:        logger.debug("Job Object creation failed for Popen (%s), no auto-cleanup", exc)
179:        return proc, None
180:
181:
182:def close_job(job_handle):
183:    """Close a Job Object handle.
184:
185:    Safe to call after the child has already been terminated — closing the
186:    handle on a dead process is a no-op. Call this in your explicit stop
187:    function after terminating the child.
188:    """
189:    if job_handle is None:
190:        return
191:    try:
192:        import ctypes
193:        ctypes.windll.kernel32.CloseHandle(job_handle)
194:    except Exception:
195:        pass
196:
197:
198:def kill_orphaned_by_cmdline(pattern, exclude_pid=None):
199:    """Find and kill processes whose command line contains *pattern*.
200:
201:    Used at startup to sweep orphaned subprocesses from a previous crash.
202:    Skips the current process and *exclude_pid* if given.
203:
204:    Returns the number of processes killed. Returns 0 on non-Windows.
205:    """
206:    if sys.platform != "win32":
207:        return 0
208:
209:    my_pid = __import__("os").getpid()
210:    skip = {my_pid}
211:    if exclude_pid is not None:
212:        skip.add(exclude_pid)
213:
214:    try:
215:        result = subprocess.run(
216:            ["wmic", "process", "where",
217:             f"CommandLine like '%{pattern}%'",
218:             "get", "ProcessId", "/format:csv"],
219:            capture_output=True, text=True, timeout=15,
220:        )
221:    except Exception as exc:
222:        logger.debug("WMIC process query failed: %s", exc)
223:        return 0
224:
225:    killed = 0
226:    for line in result.stdout.splitlines():
227:        parts = line.strip().split(",")
228:        if len(parts) < 2:
229:            continue
230:        try:
231:            pid = int(parts[-1])
232:        except ValueError:
233:            continue
234:        if pid in skip or pid == 0:
235:            continue
236:
237:        logger.info("Killing orphaned process (pattern=%r): PID %d", pattern, pid)
238:        try:
239:            subprocess.run(
240:                ["taskkill", "/F", "/PID", str(pid)],
241:                capture_output=True, timeout=10,
242:            )
243:            killed += 1
244:        except Exception:
245:            pass
246:
247:    return killed
248:
249:
250:def kill_orphaned_llama():
251:    """Find and kill orphaned llama-completion.exe processes.
252:
253:    An orphaned process is one whose parent PID no longer exists.
254:    Uses PowerShell + Win32 API to enumerate and check processes.
255:
256:    Returns the number of processes killed.  Returns 0 on non-Windows.
257:    """
258:    if sys.platform != "win32":
259:        return 0
260:
261:    import ctypes
262:
263:    kernel32 = ctypes.windll.kernel32
264:    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
265:    PROCESS_TERMINATE = 0x0001
266:
267:    # --- Get llama-completion.exe processes via PowerShell --------------------
268:    ps_cmd = (
269:        'powershell.exe -NoProfile -Command "'
270:        "Get-CimInstance Win32_Process -Filter \\\"Name='llama-completion.exe'\\\" "
271:        '| Select-Object ProcessId,ParentProcessId | ConvertTo-Json"'
272:    )
273:
274:    try:
275:        result = subprocess.run(
276:            ps_cmd,
277:            capture_output=True,
278:            text=True,
279:            timeout=15,
280:            shell=True,
281:        )
282:    except Exception as exc:
283:        logger.debug("PowerShell process query failed: %s", exc)
284:        return 0
285:
286:    if result.returncode != 0 or not result.stdout.strip():
287:        return 0
288:
289:    try:
290:        data = json.loads(result.stdout.strip())
291:    except json.JSONDecodeError:
292:        logger.debug("Failed to parse PowerShell JSON output")
293:        return 0
294:
295:    # PowerShell returns a single object (not array) when there's only one match
296:    if isinstance(data, dict):
297:        data = [data]
298:    if not isinstance(data, list):
299:        return 0
300:
301:    killed = 0
302:    for entry in data:
303:        pid = entry.get("ProcessId")
304:        ppid = entry.get("ParentProcessId")
305:        if pid is None or ppid is None:
306:            continue
307:
308:        # Check if parent is alive
309:        parent_alive = False
310:        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(ppid))
311:        if handle:
312:            parent_alive = True
313:            kernel32.CloseHandle(handle)
314:
315:        if not parent_alive:
316:            logger.info("Orphaned llama-completion.exe PID %d (parent %d dead) — killing", pid, ppid)
317:            # Terminate the orphan
318:            h_proc = kernel32.OpenProcess(PROCESS_TERMINATE, False, int(pid))
319:            if h_proc:
320:                kernel32.TerminateProcess(h_proc, 1)
321:                kernel32.CloseHandle(h_proc)
322:                killed += 1
323:                logger.info("Killed orphaned llama-completion.exe PID %d", pid)
324:            else:
325:                logger.warning("Could not open llama-completion.exe PID %d for termination", pid)
326:
327:    return killed
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 791ms:
1:"""Config validation for portfolio system startup.
2:
3:Validates config.json has all required keys before the main loop starts.
4:"""
5:
6:import logging
7:from pathlib import Path
8:
9:from portfolio.file_utils import load_json
10:
11:logger = logging.getLogger("portfolio.config_validator")
12:
13:CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.json"
14:
15:# Required: missing any of these is a fatal error.
16:# 2026-05-11 fix: Binance creds actually live under ``exchange.key`` /
17:# ``exchange.secret`` (freqtrade-style config layout, with
18:# ``exchange.name = "binance"``) — they have never been under a
19:# ``binance`` top-level section. The original list referencing
20:# ``binance.key`` / ``binance.secret`` was blocking main-loop startup on
21:# fresh restarts because validation always failed against the real
22:# config shape.
23:REQUIRED_KEYS = [
24:    ("telegram", "token"),
25:    ("telegram", "chat_id"),
26:    ("alpaca", "key"),
27:    ("alpaca", "secret"),
28:    ("exchange", "key"),
29:    ("exchange", "secret"),
30:]
31:
32:# Optional: missing these produces a warning but isn't fatal
33:OPTIONAL_KEYS = [
34:    ("mistral_api_key",),
35:    ("iskbets",),
36:    ("newsapi_key",),
37:    ("alpha_vantage", "api_key"),
38:    ("golddigger", "fred_api_key"),
39:    ("bgeometrics", "api_token"),
40:]
41:
42:
43:def validate_config(config: dict) -> list[str]:
44:    """Validate config dict. Returns list of error strings (empty = valid)."""
45:    errors = []
46:    for key_path in REQUIRED_KEYS:
47:        obj = config
48:        for key in key_path:
49:            if not isinstance(obj, dict) or key not in obj:
50:                errors.append(f"missing required key: {'.'.join(key_path)}")
51:                break
52:            obj = obj[key]
53:        else:
54:            # Key exists — check it's not empty/placeholder
55:            if isinstance(obj, str) and not obj.strip():
56:                errors.append(f"empty value for required key: {'.'.join(key_path)}")
57:    return errors
58:
59:
60:def validate_config_file() -> dict:
61:    """Load config.json, validate, and return it.
62:
63:    Logs warnings for missing optional keys.
64:    Raises ValueError if required keys are missing.
65:    """
66:    config = load_json(CONFIG_FILE)
67:    if config is None:
68:        raise ValueError(f"config.json not found or unreadable at {CONFIG_FILE}")
69:
70:    # Check optional keys and warn
71:    for key_path in OPTIONAL_KEYS:
72:        obj = config
73:        for key in key_path:
74:            if not isinstance(obj, dict) or key not in obj:
75:                logger.warning("optional config key missing: %s", '.'.join(key_path))
76:                break
77:            obj = obj[key]
78:
79:    # Check required keys
80:    errors = validate_config(config)
81:    if errors:
82:        for err in errors:
83:            logger.error("config validation: %s", err)
84:        raise ValueError(f"config.json validation failed: {'; '.join(errors)}")
85:
86:    logger.info("config.json validated successfully")
87:    return config
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
Core primitives are readable now. I’ve already got a few likely issues to verify, but I’m not ranking them until I trace their callers and the dashboard/script surfaces that can actually trigger them.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "''^" portfolio/notification_text.py' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "''^" portfolio/message_store.py' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "''^" portfolio/shadow_registry.py' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "''^" portfolio/vector_memory.py' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "''^" portfolio/backtester.py' in Q:\finance-analyzer
 succeeded in 760ms:
1:"""Shared text helpers for human-readable notifications."""
2:
3:from __future__ import annotations
4:
5:_THESIS_STATUS_LABELS = {
6:    "THREATENED": "Thesis threatened",
7:    "MIXED": "Thesis mixed",
8:    "INTACT": "Thesis intact",
9:    "NEUTRAL": "Thesis neutral",
10:}
11:
12:
13:def humanize_ticker(ticker: str | None) -> str:
14:    text = str(ticker or "")
15:    return text.replace("-USD", "").replace("-", " ").strip()
16:
17:
18:def humanize_thesis_status(status: str | None) -> str:
19:    key = str(status or "").upper()
20:    return _THESIS_STATUS_LABELS.get(key, str(status or "").replace("_", " ").title())
21:
22:
23:def format_vote_summary(buy_count: int, sell_count: int, hold_count: int | None = None) -> str:
24:    parts = [f"{int(buy_count)} buy", f"{int(sell_count)} sell"]
25:    if hold_count is not None:
26:        parts.append(f"{int(hold_count)} hold")
27:    return " / ".join(parts)
28:
29:
30:def format_confidence(confidence: float | int | None) -> str:
31:    if confidence is None:
32:        return ""
33:    return f"{int(round(float(confidence) * 100))}% confidence"
34:
35:
36:def format_fear_greed(value: int | float | None) -> str:
37:    if value is None:
38:        return ""
39:    if isinstance(value, float) and value.is_integer():
40:        value = int(value)
41:    return f"Fear & Greed {value}"
42:
43:
44:def format_portfolio_context(
45:    patient_total: float,
46:    patient_pnl: float,
47:    bold_total: float,
48:    bold_pnl: float,
49:    *,
50:    bold_holdings: str = "",
51:    consensus_accuracy: float | None = None,
52:) -> str:
53:    parts = [
54:        f"Patient portfolio {patient_total / 1000:.0f}K SEK ({patient_pnl:+.0f}%)",
55:        f"Bold portfolio {bold_total / 1000:.0f}K SEK ({bold_pnl:+.0f}%){bold_holdings}",
56:    ]
57:    if consensus_accuracy is not None:
58:        parts.append(f"Consensus accuracy {int(round(consensus_accuracy * 100))}%")
59:    return "_" + " · ".join(parts) + "_"
60:
61:
62:def format_tier_footer(label: str, tier: int, check_count: int, time_label: str) -> str:
63:    return f"_{label} Tier {tier} · Check #{check_count} · {time_label}_"
64:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 774ms:
1:"""Central message routing — save all messages to JSONL, send only selected categories to Telegram.
2:
3:Categories that are ALWAYS sent to Telegram:
4:  - trade:   simulated BUY/SELL executions (Layer 2)
5:  - iskbets: intraday entry/exit alerts
6:  - bigbet:  mean-reversion BIG BET alerts
7:  - digest:  4-hourly activity report
8:
9:Categories that are SENT to Telegram:
10:  - analysis:   HOLD analysis, market commentary (Layer 2 — sole Telegram sender)
11:
12:Categories that are ALSO SENT to Telegram:
13:  - invocation:  "Layer 2 Tx invoked" notifications
14:  - regime:      regime shift alerts
15:  - error:       loop crash notifications
16:
17:Categories that are SAVED ONLY (viewable on dashboard / via file):
18:  - fx_alert:    FX rate staleness warnings
19:"""
20:
21:import logging
22:import os
23:import re
24:from datetime import UTC, datetime
25:from pathlib import Path
26:
27:from portfolio.file_utils import atomic_append_jsonl
28:from portfolio.http_retry import fetch_with_retry
29:
30:logger = logging.getLogger("portfolio.message_store")
31:
32:BASE_DIR = Path(__file__).resolve().parent.parent
33:MESSAGES_FILE = BASE_DIR / "data" / "telegram_messages.jsonl"
34:
35:_TELEGRAM_MAX_LENGTH = 4096
36:_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
37:_COMMON_MOJIBAKE_REPLACEMENTS = {
38:    "Â·": "·",
39:    "â": "—",
40:    "â€“": "–",
41:    "â": "'",
42:    "â": "'",
43:    'â': '"',
44:    'â': '"',
45:    "â": "→",
46:    "â": "↑",
47:    "â": "↓",
48:    "Â": "",
49:}
50:
51:# Categories whose messages should be sent to Telegram in addition to being saved.
52:SEND_CATEGORIES = {"trade", "iskbets", "bigbet", "digest", "daily_digest", "analysis", "invocation", "regime", "error", "elongir", "crypto_report"}
53:
54:
55:def _repair_common_mojibake(text):
56:    repaired = text
57:    for bad, good in _COMMON_MOJIBAKE_REPLACEMENTS.items():
58:        repaired = repaired.replace(bad, good)
59:    return repaired
60:
61:
62:def _normalize_message_whitespace(text):
63:    lines = []
64:    for raw_line in text.split("\n"):
65:        if raw_line.startswith("`") and raw_line.endswith("`"):
66:            lines.append(raw_line.rstrip())
67:            continue
68:        line = raw_line.replace("\t", " ")
69:        line = re.sub(r" {2,}", " ", line).strip()
70:        lines.append(line)
71:    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
72:
73:
74:def sanitize_message_text(text):
75:    """Normalize message text before saving/sending.
76:
77:    Keeps intended Markdown structure while removing common control-byte and
78:    mojibake artifacts that make Telegram messages unreadable.
79:    """
80:    cleaned = str(text or "")
81:    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
82:    cleaned = _repair_common_mojibake(cleaned)
83:    cleaned = _CONTROL_CHAR_RE.sub(" ", cleaned)
84:    return _normalize_message_whitespace(cleaned)
85:
86:
87:def log_message(text, category="analysis", sent=False):
88:    """Append a message to the JSONL message log.
89:
90:    Args:
91:        text: Message text (may contain Markdown).
92:        category: Message category (see module docstring for valid values).
93:        sent: Whether the message was actually sent to Telegram.
94:    """
95:    cleaned = sanitize_message_text(text)
96:    entry = {
97:        "ts": datetime.now(UTC).isoformat(),
98:        "text": cleaned,
99:        "category": category,
100:        "sent": sent,
101:    }
102:    atomic_append_jsonl(MESSAGES_FILE, entry)
103:
104:
105:def _do_send_telegram(msg, config):
106:    """Actually send a message to Telegram. Returns True on success.
107:
108:    This is the raw API call — no gating by layer1_messages or category.
109:    Handles truncation, Markdown fallback on parse errors.
110:    """
111:    if os.environ.get("NO_TELEGRAM"):
112:        logger.info("[NO_TELEGRAM] Skipping send")
113:        return True
114:
115:    msg = sanitize_message_text(msg)
116:
117:    token = config.get("telegram", {}).get("token")
118:    chat_id = config.get("telegram", {}).get("chat_id")
119:    if not token or not chat_id:
120:        logger.warning("Telegram token/chat_id not configured")
121:        return False
122:
123:    # Truncate to Telegram's max message length (BUG-131: truncate at line
124:    # boundary to avoid breaking Markdown formatting mid-tag)
125:    if len(msg) > _TELEGRAM_MAX_LENGTH:
126:        logger.warning(
127:            "Telegram message truncated from %d to %d chars",
128:            len(msg), _TELEGRAM_MAX_LENGTH,
129:        )
130:        cut = _TELEGRAM_MAX_LENGTH - 20
131:        # Find last newline before cut point to avoid splitting Markdown tags
132:        nl_pos = msg.rfind("\n", 0, cut)
133:        if nl_pos > cut // 2:
134:            cut = nl_pos
135:        msg = msg[:cut] + "\n...(truncated)"
136:
137:    r = fetch_with_retry(
138:        f"https://api.telegram.org/bot{token}/sendMessage",
139:        method="POST",
140:        json_body={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
141:        timeout=30,
142:    )
143:    if r is None:
144:        return False
145:    if r.ok:
146:        return True
147:
148:    # Markdown parse failure (HTTP 400) — retry without parse_mode
149:    if r.status_code == 400:
150:        err_desc = ""
151:        try:
152:            err_desc = r.json().get("description", "")
153:        except Exception as e:
154:            logger.debug("Failed to parse Telegram error response: %s", e)
155:        if any(kw in err_desc.lower() for kw in ("parse", "markdown", "entity")):
156:            logger.warning(
157:                "Telegram Markdown parse failed (%s), resending without formatting",
158:                err_desc,
159:            )
160:            r2 = fetch_with_retry(
161:                f"https://api.telegram.org/bot{token}/sendMessage",
162:                method="POST",
163:                json_body={"chat_id": chat_id, "text": msg},
164:                timeout=30,
165:            )
166:            return r2 is not None and r2.ok
167:    return False
168:
169:
170:def send_or_store(msg, config, category="analysis"):
171:    """Central routing: save message to JSONL, optionally send to Telegram.
172:
173:    If category is in SEND_CATEGORIES, the message is sent to Telegram AND logged.
174:    Otherwise it is only logged (saved to JSONL for dashboard / file reading).
175:
176:    This function bypasses the ``layer1_messages`` config gate — the category
177:    determines whether to send, not the global flag.
178:
179:    Args:
180:        msg: Message text (may contain Markdown).
181:        config: Full config dict (needs ``telegram.token`` and ``telegram.chat_id``).
182:        category: Message category string.
183:
184:    Returns:
185:        True if message was sent (or save-only succeeded), False on send failure.
186:    """
187:    cleaned = sanitize_message_text(msg)
188:    should_send = category in SEND_CATEGORIES
189:
190:    # Mute gates: skip Telegram send, still log to JSONL
191:    tg_cfg = config.get("telegram", {})
192:
193:    # Per-category blocklist: mute specific categories
194:    muted = set(tg_cfg.get("muted_categories", []))
195:    if category in muted:
196:        log_message(cleaned, category=category, sent=False)
197:        logger.info("Message muted [%s]: %.60s...", category, cleaned.replace("\n", " "))
198:        return True
199:
200:    # Global mute gate: skip Telegram send unless category is whitelisted
201:    if tg_cfg.get("mute_all", False):
202:        unmuted = set(tg_cfg.get("unmuted_categories", []))
203:        if category not in unmuted:
204:            log_message(cleaned, category=category, sent=False)
205:            logger.info("Message muted [%s]: %.60s...", category, cleaned.replace("\n", " "))
206:            return True
207:
208:    if should_send:
209:        sent_ok = _do_send_telegram(cleaned, config)
210:        log_message(cleaned, category=category, sent=sent_ok)
211:        if sent_ok:
212:            logger.info("Message sent [%s]: %.60s...", category, cleaned.replace("\n", " "))
213:        else:
214:            logger.warning("Message send failed [%s]: %.60s...", category, cleaned.replace("\n", " "))
215:        return sent_ok
216:    else:
217:        log_message(cleaned, category=category, sent=False)
218:        logger.debug("Message stored [%s]: %.60s...", category, cleaned.replace("\n", " "))
219:        return True
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 766ms:
1:"""Backtest: replay historical signal snapshots through old vs new consensus.
2:
3:Usage:
4:    .venv/Scripts/python.exe -m portfolio.backtester [--horizon 1d] [--days 30]
5:
6:Compares:
7:  OLD: simple majority voting (no accuracy weighting)
8:  NEW: _weighted_consensus with accuracy gate + EWMA + regime + utility + MWU
9:"""
10:
11:import argparse
12:import logging
13:from datetime import UTC, datetime, timedelta
14:
15:logger = logging.getLogger("portfolio.backtester")
16:
17:HORIZONS = ["3h", "4h", "12h", "1d", "3d", "5d", "10d"]
18:
19:
20:def _old_consensus(votes):
21:    """Original simple majority voting (no weighting)."""
22:    buy = sum(1 for v in votes.values() if v == "BUY")
23:    sell = sum(1 for v in votes.values() if v == "SELL")
24:    if buy > sell:
25:        return "BUY"
26:    elif sell > buy:
27:        return "SELL"
28:    return "HOLD"
29:
30:
31:def _build_accuracy_data(horizon="1d"):
32:    """Build blended accuracy_data dict matching signal_engine's EWMA blend.
33:
34:    ARCH-23: Uses shared blend_accuracy_data() from accuracy_stats to avoid
35:    duplicating the blending logic.
36:    """
37:    from portfolio.accuracy_stats import (
38:        blend_accuracy_data,
39:        load_cached_accuracy,
40:        signal_accuracy,
41:        signal_accuracy_recent,
42:        write_accuracy_cache,
43:    )
44:
45:    acc_horizon = horizon if horizon in ("3h", "4h", "12h") else "1d"
46:
47:    alltime = load_cached_accuracy(acc_horizon)
48:    if not alltime:
49:        alltime = signal_accuracy(acc_horizon)
50:        if alltime:
51:            write_accuracy_cache(acc_horizon, alltime)
52:
53:    recent = load_cached_accuracy(f"{acc_horizon}_recent")
54:    if not recent:
55:        recent = signal_accuracy_recent(acc_horizon, days=7)
56:        if recent:
57:            write_accuracy_cache(f"{acc_horizon}_recent", recent)
58:
59:    return blend_accuracy_data(alltime, recent)
60:
61:
62:def run_backtest(horizon="1d", days=None):
63:    """Run the full backtest comparing old vs new consensus.
64:
65:    Args:
66:        horizon: Outcome horizon to evaluate.
67:        days: Lookback window in days (None = all available data).
68:
69:    Returns:
70:        dict with per-horizon and per-signal comparison results.
71:    """
72:    from portfolio.accuracy_stats import _vote_correct, load_entries
73:    from portfolio.signal_engine import (
74:        ACCURACY_GATE_THRESHOLD,
75:        _weighted_consensus,
76:    )
77:    from portfolio.tickers import DISABLED_SIGNALS, SIGNAL_NAMES
78:
79:    entries = load_entries()
80:
81:    # Apply time filter if requested
82:    cutoff = None
83:    if days is not None:
84:        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
85:
86:    if cutoff:
87:        entries = [e for e in entries if e.get("ts", "") >= cutoff]
88:
89:    if not entries:
90:        return {"error": "No entries found", "entries": 0}
91:
92:    # Pre-build accuracy_data for new consensus (based on the target horizon)
93:    accuracy_data = _build_accuracy_data(horizon)
94:
95:    # Collect results per horizon and per signal (1d only for signal breakdown)
96:    # Structure: {horizon: {old_correct, old_total, new_correct, new_total}}
97:    hz_results = {
98:        h: {"old_correct": 0, "old_total": 0, "new_correct": 0, "new_total": 0}
99:        for h in HORIZONS
100:    }
101:
102:    # Per-signal breakdown (1d horizon only for brevity)
103:    sig_results = {
104:        s: {"old_correct": 0, "old_total": 0, "new_correct": 0, "new_total": 0}
105:        for s in SIGNAL_NAMES
106:        if s not in DISABLED_SIGNALS
107:    }
108:
109:    # Track date range
110:    min_ts = None
111:    max_ts = None
112:
113:    for entry in entries:
114:        ts = entry.get("ts", "")
115:        if ts:
116:            if min_ts is None or ts < min_ts:
117:                min_ts = ts
118:            if max_ts is None or ts > max_ts:
119:                max_ts = ts
120:
121:        outcomes = entry.get("outcomes", {})
122:        tickers_data = entry.get("tickers", {})
123:
124:        for ticker, tdata in tickers_data.items():
125:            votes = tdata.get("signals", {})
126:            if not votes:
127:                continue
128:
129:            # Filter out disabled signals for old consensus too
130:            active_votes = {k: v for k, v in votes.items() if k not in DISABLED_SIGNALS}
131:            if not active_votes:
132:                continue
133:
134:            # Regime for new consensus (may not exist in old entries)
135:            regime = tdata.get("regime", "unknown")
136:
137:            # Compute old consensus (simple majority)
138:            old_action = _old_consensus(active_votes)
139:
140:            # Compute new consensus (weighted)
141:            new_action, _ = _weighted_consensus(
142:                active_votes,
143:                accuracy_data,
144:                regime,
145:                activation_rates=None,
146:                accuracy_gate=ACCURACY_GATE_THRESHOLD,
147:            )
148:
149:            # Evaluate against each horizon
150:            for h in HORIZONS:
151:                outcome = outcomes.get(ticker, {}).get(h)
152:                if not outcome:
153:                    continue
154:                change_pct = outcome.get("change_pct", 0)
155:
156:                if old_action != "HOLD":
157:                    result_val = _vote_correct(old_action, change_pct)
158:                    if result_val is not None:
159:                        hz_results[h]["old_total"] += 1
160:                        if result_val:
161:                            hz_results[h]["old_correct"] += 1
162:
163:                if new_action != "HOLD":
164:                    result_val = _vote_correct(new_action, change_pct)
165:                    if result_val is not None:
166:                        hz_results[h]["new_total"] += 1
167:                        if result_val:
168:                            hz_results[h]["new_correct"] += 1
169:
170:            # Per-signal accuracy on target horizon
171:            target_outcome = outcomes.get(ticker, {}).get(horizon)
172:            if target_outcome:
173:                change_pct = target_outcome.get("change_pct", 0)
174:                for sig_name in sig_results:
175:                    vote = votes.get(sig_name, "HOLD")
176:                    if vote == "HOLD":
177:                        continue
178:                    result_val = _vote_correct(vote, change_pct)
179:                    if result_val is None:
180:                        continue
181:
182:                    # Old: always counts this signal's vote
183:                    sig_results[sig_name]["old_total"] += 1
184:                    if result_val:
185:                        sig_results[sig_name]["old_correct"] += 1
186:
187:                    # New: only counts if signal is not gated
188:                    sig_data = accuracy_data.get(sig_name, {})
189:                    sig_acc = sig_data.get("accuracy", 0.5)
190:                    sig_samples = sig_data.get("total", 0)
191:                    from portfolio.signal_engine import ACCURACY_GATE_MIN_SAMPLES
192:                    if sig_samples >= ACCURACY_GATE_MIN_SAMPLES and sig_acc < ACCURACY_GATE_THRESHOLD:
193:                        pass  # gated out — doesn't count in new system
194:                    else:
195:                        sig_results[sig_name]["new_total"] += 1
196:                        if result_val:
197:                            sig_results[sig_name]["new_correct"] += 1
198:
199:    return {
200:        "entries": len(entries),
201:        "min_ts": min_ts,
202:        "max_ts": max_ts,
203:        "horizon_results": hz_results,
204:        "signal_results": sig_results,
205:    }
206:
207:
208:def _safe_div(a, b, default=0.0):
209:    return a / b if b > 0 else default
210:
211:
212:def print_report(results, target_horizon="1d"):
213:    """Print a formatted comparison report."""
214:    if "error" in results:
215:        print(f"Error: {results['error']}")
216:        return
217:
218:    entries = results["entries"]
219:    min_ts = results.get("min_ts", "?")
220:    max_ts = results.get("max_ts", "?")
221:
222:    # Parse date range
223:    try:
224:        d1 = datetime.fromisoformat(min_ts).date()
225:        d2 = datetime.fromisoformat(max_ts).date()
226:        days_span = (d2 - d1).days
227:        period_str = f"{d1} to {d2} ({days_span} days, {entries} entries)"
228:    except Exception:
229:        period_str = f"{entries} entries"
230:
231:    print("=== BACKTEST: Old vs New Consensus ===")
232:    print(f"Period: {period_str}")
233:    print()
234:
235:    # Per-horizon table
236:    hz_results = results["horizon_results"]
237:    print(f"{'Horizon':<10} {'Old Acc':>9} {'New Acc':>9} {'Delta':>8} {'Old N':>8} {'New N':>8}")
238:    print(f"{'-------':<10} {'-------':>9} {'-------':>9} {'-----':>8} {'-----':>8} {'-----':>8}")
239:    for h in HORIZONS:
240:        r = hz_results[h]
241:        old_n = r["old_total"]
242:        new_n = r["new_total"]
243:        if old_n == 0 and new_n == 0:
244:            continue
245:        old_acc = _safe_div(r["old_correct"], old_n)
246:        new_acc = _safe_div(r["new_correct"], new_n)
247:        delta = new_acc - old_acc
248:        marker = " *" if h == target_horizon else ""
249:        print(
250:            f"{h + marker:<10} {old_acc*100:>8.1f}% {new_acc*100:>8.1f}% {delta*100:>+7.1f}% {old_n:>8,} {new_n:>8,}"
251:        )
252:
253:    print()
254:    print(f"Per-Signal Accuracy ({target_horizon} horizon):")
255:    print(f"{'Signal':<22} {'Old':>8} {'New':>8} {'Delta':>8} {'Old N':>7} {'New N':>7}")
256:    print(f"{'------':<22} {'---':>8} {'---':>8} {'-----':>8} {'-----':>7} {'-----':>7}")
257:
258:    sig_results = results["signal_results"]
259:    # Sort by new accuracy desc
260:    sig_sorted = sorted(
261:        sig_results.items(),
262:        key=lambda x: _safe_div(x[1]["new_correct"], x[1]["new_total"]),
263:        reverse=True,
264:    )
265:    for sig_name, r in sig_sorted:
266:        old_n = r["old_total"]
267:        new_n = r["new_total"]
268:        if old_n < 5 and new_n < 5:
269:            continue  # skip signals with no data
270:        old_acc = _safe_div(r["old_correct"], old_n)
271:        # Signal gated in new system when it had old data but new_n == 0
272:        if old_n >= 5 and new_n == 0:
273:            new_acc_str = "  GATED"
274:            delta_str = "       "
275:        else:
276:            new_acc = _safe_div(r["new_correct"], new_n)
277:            delta = new_acc - old_acc
278:            new_acc_str = f"{new_acc*100:>7.1f}%"
279:            delta_str = f"{delta*100:>+7.1f}%"
280:        print(
281:            f"{sig_name:<22} {old_acc*100:>7.1f}% {new_acc_str} {delta_str} {old_n:>7,} {new_n:>7,}"
282:        )
283:
284:    # Summary
285:    print()
286:    r_target = hz_results.get(target_horizon, {})
287:    old_acc_t = _safe_div(r_target.get("old_correct", 0), r_target.get("old_total", 0))
288:    new_acc_t = _safe_div(r_target.get("new_correct", 0), r_target.get("new_total", 0))
289:    delta_t = new_acc_t - old_acc_t
290:    print(f"=== Summary ({target_horizon}) ===")
291:    print(f"Old system accuracy: {old_acc_t*100:.1f}% ({r_target.get('old_total', 0):,} samples)")
292:    print(f"New system accuracy: {new_acc_t*100:.1f}% ({r_target.get('new_total', 0):,} samples)")
293:    print(f"Improvement: {delta_t*100:+.1f}pp")
294:
295:
296:if __name__ == "__main__":
297:    logging.basicConfig(level=logging.WARNING)
298:
299:    parser = argparse.ArgumentParser(
300:        description="Backtest old vs new consensus systems on historical signal data."
301:    )
302:    parser.add_argument(
303:        "--horizon",
304:        default="1d",
305:        choices=HORIZONS,
306:        help="Primary horizon to evaluate (default: 1d)",
307:    )
308:    parser.add_argument(
309:        "--days",
310:        type=int,
311:        default=None,
312:        help="Lookback window in days (default: all available data)",
313:    )
314:    args = parser.parse_args()
315:
316:    results = run_backtest(horizon=args.horizon, days=args.days)
317:    print_report(results, target_horizon=args.horizon)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 770ms:
1:"""Shadow-signal registry with promotion criteria and age tracking.
2:
3:Purpose
4:-------
5:A signal enters "shadow" mode when its output is logged but not voted —
6:typically because we want to accumulate ground-truth data before trusting
7:its votes. Without explicit tracking, signals get forgotten in shadow for
8:months (FinGPT sat ~3 weeks without a single accuracy measurement, Kronos
9:ran 3668 predictions in shadow mode that all collapsed to HOLD). This
10:registry records:
11:
12:* When each signal entered shadow.
13:* What promotion criteria were agreed.
14:* When it was last reviewed.
15:* A resolution outcome when it exits (promoted / retired / still-shadow).
16:
17:The registry is a plain JSON file — no DB dependency. `scripts/review_shadow_signals.py`
18:reads it and flags any shadow older than 30 days without a resolution.
19:
20:Schema
21:------
22:```json
23:{
24:  "shadows": {
25:    "fingpt": {
26:      "entered_shadow_ts": "2026-04-09T00:00:00+00:00",
27:      "promotion_criteria": {
28:        "min_samples": 200,
29:        "min_accuracy": 0.60,
30:        "max_missing_outcome_rate": 0.20
31:      },
32:      "last_reviewed_ts": "2026-04-21T13:45:00+00:00",
33:      "status": "shadow",
34:      "notes": "Parser fix shipped 2026-04-09; outcome backfill pending."
35:    }
36:  }
37:}
38:```
39:
40:`status` is one of: `"shadow"`, `"promoted"`, `"retired"`.
41:"""
42:
43:from __future__ import annotations
44:
45:import datetime as _dt
46:import logging
47:from pathlib import Path
48:
49:from portfolio.file_utils import atomic_write_json, load_json
50:
51:logger = logging.getLogger("portfolio.shadow_registry")
52:
53:_BASE_DIR = Path(__file__).resolve().parent.parent
54:_REGISTRY_FILE = _BASE_DIR / "data" / "shadow_registry.json"
55:_STALE_DAYS = 30
56:
57:_VALID_STATUS = frozenset({"shadow", "promoted", "retired"})
58:
59:
60:def _now() -> str:
61:    return _dt.datetime.now(_dt.UTC).isoformat()
62:
63:
64:def load_registry(path: Path | str | None = None) -> dict:
65:    """Load the registry. Returns `{"shadows": {}}` when the file is
66:    missing or malformed (never raises)."""
67:    p = Path(path) if path else _REGISTRY_FILE
68:    data = load_json(str(p), default=None)
69:    if not isinstance(data, dict) or "shadows" not in data:
70:        return {"shadows": {}}
71:    return data
72:
73:
74:def save_registry(data: dict, path: Path | str | None = None) -> None:
75:    """Atomically write the registry."""
76:    p = Path(path) if path else _REGISTRY_FILE
77:    p.parent.mkdir(parents=True, exist_ok=True)
78:    atomic_write_json(str(p), data)
79:
80:
81:def add_shadow(
82:    signal: str,
83:    promotion_criteria: dict,
84:    notes: str = "",
85:    *,
86:    entered_ts: str | None = None,
87:    path: Path | str | None = None,
88:) -> None:
89:    """Register `signal` as entering shadow. If already present, update
90:    `promotion_criteria` and `notes`, reset status to `"shadow"`, and
91:    refresh `last_reviewed_ts` — but PRESERVE `entered_shadow_ts` so
92:    days-in-shadow accounting survives re-registration."""
93:    reg = load_registry(path=path)
94:    now = _now()
95:    existing = reg["shadows"].get(signal, {})
96:    entered = entered_ts or existing.get("entered_shadow_ts") or now
97:    reg["shadows"][signal] = {
98:        "entered_shadow_ts": entered,
99:        "promotion_criteria": dict(promotion_criteria),
100:        "last_reviewed_ts": now,
101:        "status": "shadow",
102:        "notes": notes or existing.get("notes", ""),
103:    }
104:    save_registry(reg, path=path)
105:
106:
107:def resolve_shadow(
108:    signal: str,
109:    status: str,
110:    notes: str = "",
111:    *,
112:    path: Path | str | None = None,
113:) -> bool:
114:    """Mark a shadow as promoted or retired. Returns True if found, False
115:    otherwise. Does NOT delete the entry — keeps resolution history."""
116:    if status not in _VALID_STATUS:
117:        raise ValueError(f"status must be in {_VALID_STATUS}, got {status!r}")
118:    reg = load_registry(path=path)
119:    entry = reg["shadows"].get(signal)
120:    if entry is None:
121:        return False
122:    entry["status"] = status
123:    entry["last_reviewed_ts"] = _now()
124:    if notes:
125:        entry["notes"] = notes
126:    save_registry(reg, path=path)
127:    return True
128:
129:
130:def days_in_shadow(signal: str, *, path: Path | str | None = None,
131:                    now: _dt.datetime | None = None) -> float | None:
132:    """Return days elapsed since signal entered shadow. None if unknown."""
133:    reg = load_registry(path=path)
134:    entry = reg["shadows"].get(signal)
135:    if entry is None:
136:        return None
137:    entered_raw = entry.get("entered_shadow_ts")
138:    if not entered_raw:
139:        return None
140:    try:
141:        entered = _dt.datetime.fromisoformat(entered_raw)
142:    except (TypeError, ValueError):
143:        return None
144:    if entered.tzinfo is None:
145:        entered = entered.replace(tzinfo=_dt.UTC)
146:    cur = now or _dt.datetime.now(_dt.UTC)
147:    return (cur - entered).total_seconds() / 86400.0
148:
149:
150:def stale_shadows(*, stale_days: int = _STALE_DAYS,
151:                   path: Path | str | None = None,
152:                   now: _dt.datetime | None = None) -> list[dict]:
153:    """Return shadow entries that are still `"shadow"` and older than
154:    `stale_days`. Each dict includes `signal`, `days_in_shadow`, and the
155:    full entry for convenience."""
156:    reg = load_registry(path=path)
157:    stale = []
158:    for sig, entry in reg["shadows"].items():
159:        if entry.get("status") != "shadow":
160:            continue
161:        age = days_in_shadow(sig, path=path, now=now)
162:        if age is None:
163:            continue
164:        if age >= stale_days:
165:            stale.append({
166:                "signal": sig,
167:                "days_in_shadow": age,
168:                **entry,
169:            })
170:    return sorted(stale, key=lambda x: -x["days_in_shadow"])
171:
172:
173:def seed_defaults(path: Path | str | None = None) -> None:
174:    """Idempotent one-time seeding for the 2026-04-21 LLM-health audit.
175:    Only adds entries that don't already exist — safe to re-run."""
176:    reg = load_registry(path=path)
177:    defaults = {
178:        "fingpt": {
179:            "entered_shadow_ts": "2026-04-09T00:00:00+00:00",
180:            "promotion_criteria": {
181:                "min_samples": 200,
182:                "min_accuracy": 0.60,
183:                "max_missing_outcome_rate": 0.20,
184:            },
185:            "notes": "Parser fix shipped 2026-04-09 (fde9cf8+28aa5d0). "
186:                     "Accuracy vs outcomes not yet measured — awaiting "
187:                     "outcome backfill for sentiment_ab_log.",
188:        },
189:        "finbert": {
190:            "entered_shadow_ts": "2026-04-09T00:00:00+00:00",
191:            "promotion_criteria": {
192:                "min_samples": 200,
193:                "min_accuracy": 0.60,
194:                "max_missing_outcome_rate": 0.20,
195:            },
196:            "notes": "CPU-cheap shadow alongside CryptoBERT/Trading-Hero-LLM. "
197:                     "86% neutral output, 87.9% primary-agreement — likely "
198:                     "collapsed to safe-label. Keep for observation only.",
199:        },
200:        "kronos": {
201:            "entered_shadow_ts": "2026-03-27T15:10:00+00:00",
202:            "promotion_criteria": {
203:                "min_samples": 500,
204:                "min_accuracy": 0.55,
205:                "min_subprocess_success_rate": 0.90,
206:            },
207:            "notes": "Un-retired 2026-04-21 afternoon with proper vote-pool "
208:                     "isolation (shadow sub-signal excluded from "
209:                     "_health_weighted_vote). Subprocess reliability still "
210:                     "59% — fix is a separate work stream.",
211:        },
212:        "credit_spread_risk": {
213:            "entered_shadow_ts": "2026-04-11T00:00:00+00:00",
214:            "promotion_criteria": {
215:                "min_samples": 200,
216:                "min_accuracy": 0.55,
217:            },
218:            "notes": "Registered but force-HOLD via DISABLED_SIGNALS pending "
219:                     "live validation.",
220:        },
221:        "crypto_macro": {
222:            "entered_shadow_ts": "2026-04-11T00:00:00+00:00",
223:            "promotion_criteria": {
224:                "min_samples": 200,
225:                "min_accuracy": 0.55,
226:            },
227:            "notes": "Registered but force-HOLD via DISABLED_SIGNALS pending "
228:                     "live validation.",
229:        },
230:    }
231:    for sig, cfg in defaults.items():
232:        if sig in reg["shadows"]:
233:            continue
234:        reg["shadows"][sig] = {
235:            **cfg,
236:            "last_reviewed_ts": _now(),
237:            "status": "shadow",
238:        }
239:    save_registry(reg, path=path)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 800ms:
1:"""ChromaDB-backed semantic memory for journal entries.
2:
3:Uses ChromaDB's built-in all-MiniLM-L6-v2 embeddings (via onnxruntime,
4:no sentence-transformers needed). Lazy-init singleton. Embeds journal
5:entries on read, queries by market state similarity.
6:
7:Config:
8:    "vector_memory": {
9:        "enabled": false,
10:        "collection": "trade_journal",
11:        "top_k": 5
12:    }
13:
14:Requires: pip install chromadb
15:Defaults to disabled — graceful fallback if chromadb is not installed.
16:"""
17:
18:import hashlib
19:import json
20:import logging
21:from pathlib import Path
22:
23:logger = logging.getLogger("portfolio.vector_memory")
24:
25:DATA_DIR = Path(__file__).resolve().parent.parent / "data"
26:CHROMADB_DIR = DATA_DIR / "chromadb"
27:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
28:
29:# Singleton
30:_collection = None
31:_client = None
32:
33:
34:def _get_collection(collection_name="trade_journal"):
35:    """Lazy-init ChromaDB client and return the collection.
36:
37:    Raises ImportError if chromadb is not installed.
38:    """
39:    global _client, _collection
40:
41:    if _collection is not None:
42:        return _collection
43:
44:    import chromadb
45:
46:    CHROMADB_DIR.mkdir(parents=True, exist_ok=True)
47:    _client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
48:    _collection = _client.get_or_create_collection(
49:        name=collection_name,
50:        metadata={"hnsw:space": "cosine"},
51:    )
52:    logger.info("ChromaDB collection '%s' ready (%d entries)",
53:                collection_name, _collection.count())
54:    return _collection
55:
56:
57:def entry_to_text(entry):
58:    """Convert a journal entry to a searchable text string.
59:
60:    Captures regime, decisions, ticker outlooks, theses, debate fields,
61:    watchlist, and reflection — everything Layer 2 would want to match on.
62:    """
63:    parts = []
64:
65:    regime = entry.get("regime", "")
66:    if regime:
67:        parts.append(f"regime: {regime}")
68:
69:    trigger = entry.get("trigger", "")
70:    if trigger:
71:        parts.append(f"trigger: {trigger}")
72:
73:    decisions = entry.get("decisions", {})
74:    for strat in ("patient", "bold"):
75:        d = decisions.get(strat, {})
76:        action = d.get("action", "HOLD")
77:        reasoning = d.get("reasoning", "")
78:        parts.append(f"{strat}: {action} — {reasoning}")
79:
80:    tickers = entry.get("tickers", {})
81:    for ticker, info in tickers.items():
82:        outlook = info.get("outlook", "neutral")
83:        thesis = info.get("thesis", "")
84:        conviction = info.get("conviction", 0)
85:        line = f"{ticker}: {outlook}"
86:        if conviction:
87:            line += f" ({conviction:.0%})"
88:        if thesis:
89:            line += f" — {thesis}"
90:        parts.append(line)
91:
92:        # Debate fields
93:        debate = info.get("debate")
94:        if debate and isinstance(debate, dict):
95:            for field in ("bull", "bear", "synthesis"):
96:                text = debate.get(field, "")
97:                if text:
98:                    parts.append(f"  {field}: {text}")
99:
100:    reflection = entry.get("reflection", "")
101:    if reflection:
102:        parts.append(f"reflection: {reflection}")
103:
104:    watchlist = entry.get("watchlist", [])
105:    if watchlist:
106:        parts.append("watchlist: " + "; ".join(watchlist))
107:
108:    return "\n".join(parts)
109:
110:
111:def _entry_id(entry):
112:    """Generate a stable ID for a journal entry based on its timestamp."""
113:    ts = entry.get("ts", "")
114:    return hashlib.md5(ts.encode()).hexdigest()
115:
116:
117:def embed_entries(entries, collection_name="trade_journal"):
118:    """Embed journal entries that aren't yet in ChromaDB.
119:
120:    Args:
121:        entries: list of journal entry dicts.
122:        collection_name: ChromaDB collection name.
123:
124:    Returns:
125:        int: number of newly embedded entries.
126:    """
127:    collection = _get_collection(collection_name)
128:
129:    # Get existing IDs
130:    existing = set()
131:    if collection.count() > 0:
132:        result = collection.get()
133:        existing = set(result["ids"])
134:
135:    new_docs = []
136:    new_ids = []
137:    new_metas = []
138:
139:    for entry in entries:
140:        eid = _entry_id(entry)
141:        if eid in existing:
142:            continue
143:        text = entry_to_text(entry)
144:        if not text.strip():
145:            continue
146:        new_docs.append(text)
147:        new_ids.append(eid)
148:        new_metas.append({
149:            "ts": entry.get("ts", ""),
150:            "regime": entry.get("regime", ""),
151:        })
152:
153:    if new_docs:
154:        collection.add(documents=new_docs, ids=new_ids, metadatas=new_metas)
155:        logger.info("Embedded %d new journal entries", len(new_docs))
156:
157:    return len(new_docs)
158:
159:
160:def query_similar(query_text, top_k=5, collection_name="trade_journal"):
161:    """Query ChromaDB for journal entries similar to query_text.
162:
163:    Args:
164:        query_text: text describing current market state.
165:        top_k: number of results.
166:        collection_name: ChromaDB collection name.
167:
168:    Returns:
169:        list of dicts with keys: text, ts, regime, distance.
170:    """
171:    collection = _get_collection(collection_name)
172:    if collection.count() == 0:
173:        return []
174:
175:    results = collection.query(
176:        query_texts=[query_text],
177:        n_results=min(top_k, collection.count()),
178:    )
179:
180:    entries = []
181:    for i, doc in enumerate(results["documents"][0]):
182:        meta = results["metadatas"][0][i] if results["metadatas"] else {}
183:        dist = results["distances"][0][i] if results["distances"] else 0
184:        entries.append({
185:            "text": doc,
186:            "ts": meta.get("ts", ""),
187:            "regime": meta.get("regime", ""),
188:            "distance": dist,
189:        })
190:
191:    return entries
192:
193:
194:def build_query_text(market_state):
195:    """Convert current market state into a query text for semantic search.
196:
197:    Args:
198:        market_state: dict with signals, held_tickers, regime, prices.
199:
200:    Returns:
201:        str: query text.
202:    """
203:    parts = []
204:
205:    regime = market_state.get("regime", "")
206:    if regime:
207:        parts.append(f"regime: {regime}")
208:
209:    held = market_state.get("held_tickers", [])
210:    if held:
211:        parts.append(f"holding: {', '.join(held)}")
212:
213:    signals = market_state.get("signals", {})
214:    for ticker, sig in signals.items():
215:        if not isinstance(sig, dict):
216:            continue
217:        action = sig.get("action", "HOLD")
218:        if action != "HOLD":
219:            conf = sig.get("confidence", 0)
220:            parts.append(f"{ticker}: {action} ({conf:.0%})")
221:
222:    return "\n".join(parts) if parts else ""
223:
224:
225:def get_semantic_context(market_state, bm25_timestamps=None,
226:                         top_k=5, collection_name="trade_journal"):
227:    """Full semantic retrieval pipeline: embed new entries, query, de-dup.
228:
229:    Args:
230:        market_state: dict with signals, held_tickers, regime, prices.
231:        bm25_timestamps: set of timestamp strings already returned by BM25
232:            (for de-duplication).
233:        top_k: number of semantic results to return.
234:        collection_name: ChromaDB collection name.
235:
236:    Returns:
237:        list of dicts with text, ts, regime, distance.
238:        Returns empty list on any error.
239:    """
240:    try:
241:        # Embed any new entries
242:        entries = _load_journal_entries()
243:        if entries:
244:            embed_entries(entries, collection_name)
245:
246:        # Build query
247:        query = build_query_text(market_state)
248:        if not query:
249:            return []
250:
251:        # Query
252:        results = query_similar(query, top_k=top_k * 2, collection_name=collection_name)
253:
254:        # De-duplicate against BM25
255:        bm25_ts = set(bm25_timestamps or [])
256:        deduped = [r for r in results if r.get("ts", "") not in bm25_ts]
257:
258:        return deduped[:top_k]
259:    except ImportError:
260:        logger.debug("chromadb not installed, skipping vector memory")
261:        return []
262:    except Exception as e:
263:        logger.warning("vector memory error: %s", e)
264:        return []
265:
266:
267:def _load_journal_entries():
268:    """Load all journal entries from JSONL."""
269:    if not JOURNAL_FILE.exists():
270:        return []
271:    entries = []
272:    with open(JOURNAL_FILE, encoding="utf-8") as f:
273:        for line in f:
274:            line = line.strip()
275:            if not line:
276:                continue
277:            try:
278:                entries.append(json.loads(line))
279:            except json.JSONDecodeError:
280:                continue
281:    return entries
282:
283:
284:def reset():
285:    """Reset the singleton (for testing)."""
286:    global _client, _collection
287:    _client = None
288:    _collection = None
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "''^" dashboard/auth.py' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "''^" dashboard/app.py' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "''^" dashboard/export_static.py' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "''^" dashboard/house_blueprint.py' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "''^" dashboard/system_status.py' in Q:\finance-analyzer
 succeeded in 1111ms:
1:"""Auth + token gating for the finance-analyzer dashboard.
2:
3:Extracted from dashboard/app.py on 2026-05-02 to break a circular import:
4:dashboard/house_blueprint.py needs `_get_config` and `require_auth`. Before
5:this split it imported them from dashboard/app.py, which imports
6:house_blueprint at module-init time. When app.py runs as __main__ (the
7:PF-Dashboard scheduled task), Python doesn't auto-register it under
8:`dashboard.app` in sys.modules — so the blueprint's `from dashboard.app
9:import ...` triggered a fresh import of the same source file, recursing
10:into a partially-loaded house_blueprint and crashing with ImportError
11:on `bp`.
12:
13:Both dashboard/app.py and dashboard/house_blueprint.py now import their
14:auth dependencies from this module. This module imports nothing
15:dashboard-specific and uses its own tiny config cache instead of
16:dashboard/app.py's main TTL cache (which would re-introduce the circle).
17:
18:Tests that patched `dashboard.app._get_dashboard_token` or
19:`dashboard.app._get_config` for auth purposes have been updated to patch
20:`dashboard.auth.*` instead, since require_auth now resolves those names
21:via its own module globals. App.py re-exports the names for backward
22:compatibility, but tests should target dashboard.auth as the canonical
23:location.
24:"""
25:from __future__ import annotations
26:
27:import functools
28:import hmac
29:import json
30:import threading
31:import time
32:from pathlib import Path
33:
34:from flask import jsonify, make_response, request
35:
36:# Path resolution mirrors dashboard/app.py — config.json sits at the repo
37:# root, two levels up from this file.
38:CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
39:
40:# Cookie auth (added 2026-04-30, rolling refresh added 2026-05-02). 365
41:# sits just under Chrome's silent 400-day cookie max-age cap (introduced
42:# 2022) — any larger value is clamped browser-side, so 365 is the effective
43:# ceiling. Combined with require_auth's per-request refresh, an active
44:# user effectively never re-authenticates.
45:COOKIE_NAME = "pf_dashboard_token"
46:COOKIE_MAX_AGE = 365 * 24 * 3600
47:
48:
49:# ---------------------------------------------------------------------------
50:# Tiny config cache — separate from dashboard/app.py's main TTL cache so
51:# this module stays self-contained.
52:# ---------------------------------------------------------------------------
53:
54:_CFG_VALUE: dict | None = None
55:_CFG_AT: float = 0.0
56:_CFG_LOCK = threading.Lock()
57:_CFG_TTL = 60.0
58:
59:
60:def _read_config_uncached() -> dict:
61:    try:
62:        with open(CONFIG_PATH, encoding="utf-8") as fp:
63:            data = json.load(fp)
64:        return data if isinstance(data, dict) else {}
65:    except (FileNotFoundError, json.JSONDecodeError, OSError):
66:        return {}
67:
68:
69:def _get_config() -> dict:
70:    """Read config.json with a 60-second in-memory cache."""
71:    global _CFG_VALUE, _CFG_AT
72:    now = time.monotonic()
73:    with _CFG_LOCK:
74:        if _CFG_VALUE is not None and (now - _CFG_AT) < _CFG_TTL:
75:            return _CFG_VALUE
76:        _CFG_VALUE = _read_config_uncached()
77:        _CFG_AT = now
78:        return _CFG_VALUE
79:
80:
81:def _get_dashboard_token() -> str | None:
82:    """Return the configured dashboard_token, or None if not set."""
83:    return _get_config().get("dashboard_token") or None
84:
85:
86:# ---------------------------------------------------------------------------
87:# Auth decorator
88:# ---------------------------------------------------------------------------
89:
90:def _refresh_cookie(response, token: str):
91:    """Refresh the auth cookie's expiry on `response`."""
92:    response.set_cookie(
93:        COOKIE_NAME,
94:        token,
95:        max_age=COOKIE_MAX_AGE,
96:        httponly=True,
97:        secure=True,
98:        samesite="Lax",
99:    )
100:    return response
101:
102:
103:def require_auth(f):
104:    """Decorator: check Cloudflare Access header, cookie, query, or bearer.
105:
106:    Validation order:
107:      0. Cf-Access-Authenticated-User-Email header — Cloudflare Access has
108:         already authenticated and policy-checked this request. Trust it.
109:      1. Cookie (`pf_dashboard_token`) — for repeat visits.
110:      2. ?token= query param — for first-visit-from-a-new-browser.
111:      3. Authorization: Bearer header — for CLI / script clients.
112:
113:    On any successful path 0-2, refreshes the cookie's 1-year expiry so
114:    it slides forward — an active user effectively never re-authenticates.
115:
116:    If no dashboard_token is configured, access is allowed (backwards
117:    compatible). Returns 401 for invalid/missing tokens.
118:    """
119:    @functools.wraps(f)
120:    def decorated(*args, **kwargs):
121:        expected = _get_dashboard_token()
122:        if expected is None:
123:            return f(*args, **kwargs)
124:
125:        # 0. Cloudflare Access — added 2026-05-02. CF strips inbound
126:        # Cf-Access-* headers at its edge and re-injects them only after
127:        # successful Access policy evaluation. Require BOTH the email
128:        # header AND the JWT assertion header — CF Access always sends
129:        # both, but a LAN attacker spoofing headers would need to know
130:        # about and forge both. Without the JWT check, any LAN client
131:        # could set the email header and bypass auth entirely.
132:        cf_email = request.headers.get("Cf-Access-Authenticated-User-Email")
133:        cf_jwt = request.headers.get("Cf-Access-Jwt-Assertion")
134:        if cf_email and cf_jwt:
135:            return _refresh_cookie(make_response(f(*args, **kwargs)), expected)
136:
137:        # 1. Cookie
138:        cookie_token = request.cookies.get(COOKIE_NAME)
139:        if cookie_token and hmac.compare_digest(cookie_token, expected):
140:            return _refresh_cookie(make_response(f(*args, **kwargs)), expected)
141:
142:        # 2. Query param
143:        token = request.args.get("token")
144:        if token and hmac.compare_digest(token, expected):
145:            return _refresh_cookie(make_response(f(*args, **kwargs)), expected)
146:
147:        # 3. Authorization: Bearer (CLI / script clients — no cookie set
148:        # since these don't usually carry one across requests anyway)
149:        auth_header = request.headers.get("Authorization", "")
150:        if auth_header.startswith("Bearer "):
151:            bearer_token = auth_header[7:].strip()
152:            if hmac.compare_digest(bearer_token, expected):
153:                return f(*args, **kwargs)
154:
155:        return jsonify({
156:            "error": "Unauthorized",
157:            "message": (
158:                "Visit /?token=YOUR_TOKEN once to set a 1-year rolling auth "
159:                "cookie. Replace YOUR_TOKEN with the dashboard_token from "
160:                "config.json. (If you arrived here through Cloudflare Access, "
161:                "this means Access didn't inject its identity header — "
162:                "contact the app owner.)"
163:            ),
164:        }), 401
165:
166:    return decorated
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 1124ms:
1:"""Portfolio Intelligence Dashboard — lightweight Flask API + frontend."""
2:
3:import functools
4:import hmac
5:import logging
6:import math
7:import threading
8:import time
9:from datetime import UTC, datetime
10:from pathlib import Path
11:from typing import Any
12:from zoneinfo import ZoneInfo
13:
14:from flask import Flask, jsonify, make_response, redirect, request, send_from_directory
15:from flask.json.provider import DefaultJSONProvider
16:
17:logger = logging.getLogger(__name__)
18:
19:
20:def _json_safe(value):
21:    """Convert NaN/Infinity to JSON-safe null recursively."""
22:    if isinstance(value, float):
23:        return value if math.isfinite(value) else None
24:    if isinstance(value, dict):
25:        return {key: _json_safe(item) for key, item in value.items()}
26:    if isinstance(value, list):
27:        return [_json_safe(item) for item in value]
28:    if isinstance(value, tuple):
29:        return [_json_safe(item) for item in value]
30:    return value
31:
32:
33:class SafeJSONProvider(DefaultJSONProvider):
34:    """Flask JSON provider that strips non-finite floats."""
35:
36:    def dumps(self, obj, **kwargs):
37:        return super().dumps(_json_safe(obj), **kwargs)
38:
39:
40:app = Flask(__name__, static_folder="static")
41:app.json = SafeJSONProvider(app)
42:
43:
44:_ALLOWED_ORIGINS = {
45:    "http://localhost:5055",
46:    "http://127.0.0.1:5055",
47:    "http://localhost:3000",
48:    "http://127.0.0.1:3000",
49:}
50:
51:
52:@app.after_request
53:def add_cors_headers(response):
54:    """Allow same-network browser access from known origins only (BUG-230)."""
55:    origin = request.headers.get("Origin", "")
56:    if origin in _ALLOWED_ORIGINS:
57:        response.headers["Access-Control-Allow-Origin"] = origin
58:    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
59:    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
60:    response.headers["Access-Control-Allow-Credentials"] = "false"
61:    return response
62:
63:DATA_DIR = Path(__file__).resolve().parent.parent / "data"
64:TRAINING_DIR = Path(__file__).resolve().parent.parent / "training" / "lora"
65:CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
66:STOCKHOLM_TZ = ZoneInfo("Europe/Stockholm")
67:
68:import sys
69:
70:sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
71:from portfolio.file_utils import load_json as _load_json_impl
72:from portfolio.file_utils import load_jsonl as _load_jsonl_impl
73:from portfolio.file_utils import load_jsonl_tail as _load_jsonl_tail_impl
74:
75:# ---------------------------------------------------------------------------
76:# TTL Cache (BUG-130: avoid re-reading files on every API request)
77:# ---------------------------------------------------------------------------
78:
79:_cache = {}
80:_cache_lock = threading.Lock()
81:_DEFAULT_TTL = 5  # seconds
82:
83:
84:def _cached_read(key, ttl, read_fn):
85:    """Return cached result if fresh, otherwise call read_fn and cache."""
86:    now = time.monotonic()
87:    with _cache_lock:
88:        entry = _cache.get(key)
89:        if entry and (now - entry[1]) < ttl:
90:            return entry[0]
91:    result = read_fn()
92:    with _cache_lock:
93:        _cache[key] = (result, now)
94:    return result
95:
96:
97:# ---------------------------------------------------------------------------
98:# Helpers
99:# ---------------------------------------------------------------------------
100:
101:def _read_json(path, ttl=_DEFAULT_TTL):
102:    return _cached_read(f"json:{path}", ttl, lambda: _load_json_impl(path))
103:
104:
105:def _read_jsonl(path, limit=100, ttl=_DEFAULT_TTL):
106:    """Cached JSONL read returning the last `limit` entries.
107:
108:    Switched from load_jsonl(limit=) (full scan + deque) to
109:    load_jsonl_tail (seek from end). For an 80MB log the difference is
110:    ~880ms vs ~5ms.
111:
112:    2026-05-04 codex P2-1 follow-up: the original 4 MB tail-bytes
113:    ceiling could silently under-deliver entries when callers ask for
114:    a large window AND individual rows are large (e.g. /api/telegrams
115:    requests 5000 entries × up to 4 KB each ≈ 20 MB needed). The
116:    fetcher now grows tail_bytes adaptively — doubling on each retry
117:    until either `limit` rows are parsed or the whole file has been
118:    pulled — and falls through to the full-scan path as a final
119:    safety net. Cache key bumped to v2 so old (potentially
120:    under-delivered) entries don't survive the deploy.
121:    """
122:    if limit and limit > 0:
123:        return _cached_read(
124:            f"jsonl_tail_v2:{path}:{limit}",
125:            ttl,
126:            lambda: _read_tail_with_growth(path, limit),
127:        )
128:    return _cached_read(
129:        f"jsonl:{path}:{limit}", ttl, lambda: _load_jsonl_impl(path, limit=limit)
130:    )
131:
132:
133:def _read_tail_with_growth(path, limit):
134:    """Read tail entries, doubling tail_bytes until we have `limit`
135:    parsed rows or the whole file has been consumed.
136:
137:    Falls back to the full-scan load_jsonl path if even reading the
138:    full file via the tail helper still yields < limit entries —
139:    that case implies the tail helper's first-line-drop heuristic is
140:    chewing through real data and we should bypass it entirely.
141:    """
142:    try:
143:        file_size = Path(path).stat().st_size
144:    except (FileNotFoundError, OSError):
145:        return []
146:    if file_size == 0:
147:        return []
148:
149:    # Initial budget: ~1 KB per entry with a 512 KB floor.
150:    tail_bytes = max(512_000, limit * 1024)
151:    # Cap retry budget at 64 MB to avoid runaway reads on a corrupt or
152:    # absurdly-sized file. Most logs in this codebase are < 100 MB and
153:    # 64 MB will hold ~64 K typical-sized entries.
154:    max_retry_bytes = 64 * 1024 * 1024
155:    while True:
156:        capped = min(tail_bytes, file_size, max_retry_bytes)
157:        rows = _load_jsonl_tail_impl(path, max_entries=limit,
158:                                       tail_bytes=capped)
159:        if len(rows) >= limit or capped >= file_size or capped >= max_retry_bytes:
160:            break
161:        tail_bytes *= 2
162:
163:    # Last-chance fallback: if even the full-file tail came up short,
164:    # the issue isn't byte budget — it's the first-line-drop heuristic.
165:    # Fall through to the canonical full-scan reader.
166:    if len(rows) < limit and capped >= file_size:
167:        rows = _load_jsonl_impl(path, limit=limit)
168:    return rows
169:
170:
171:def _get_config():
172:    return _read_json(CONFIG_PATH, ttl=60) or {}
173:
174:
175:def _parse_limit_arg(name, default, max_value):
176:    """Parse integer query arg with sane bounds and fallback."""
177:    try:
178:        value = int(request.args.get(name, default))
179:    except (ValueError, TypeError):
180:        value = default
181:    return max(1, min(value, max_value))
182:
183:
184:def _iter_latest_dict_entries(path, read_limit):
185:    """Yield JSONL entries newest-first, skipping non-dict shapes."""
186:    raw = _read_jsonl(path, limit=read_limit)
187:    for entry in reversed(raw):
188:        if isinstance(entry, dict):
189:            yield entry
190:
191:
192:def _parse_iso8601(value):
193:    """Parse an ISO-8601 timestamp into an aware datetime."""
194:    if not value or not isinstance(value, str):
195:        return None
196:    try:
197:        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
198:    except ValueError:
199:        return None
200:    if dt.tzinfo is None:
201:        dt = dt.replace(tzinfo=UTC)
202:    return dt
203:
204:
205:def _stockholm_now():
206:    return datetime.now(UTC).astimezone(STOCKHOLM_TZ)
207:
208:
209:def _hours_until_stockholm_close(now=None, close_hour=21, close_minute=30):
210:    """Return hours remaining until the Stockholm warrant close.
211:
212:    Defaults updated 2026-05-11 to match the unified 08:30–21:30
213:    trading window (previously 21:55, tracked GoldDigger's old US-overlap
214:    end). Callers that need the legacy 21:55 must pass it explicitly.
215:    """
216:    now = (now or _stockholm_now()).astimezone(STOCKHOLM_TZ)
217:    close_dt = now.replace(hour=close_hour, minute=close_minute, second=0, microsecond=0)
218:    if now >= close_dt:
219:        return 0.0
220:    return round((close_dt - now).total_seconds() / 3600.0, 2)
221:
222:
223:def _is_number(value):
224:    return isinstance(value, (int, float)) and math.isfinite(value)
225:
226:
227:def _round_or_none(value, digits=2):
228:    return round(float(value), digits) if _is_number(value) else None
229:
230:
231:def _normalize_golddigger_position(raw_position, latest_log):
232:    if not isinstance(raw_position, dict):
233:        return None
234:
235:    quantity = raw_position.get("quantity", raw_position.get("shares"))
236:    entry_price = raw_position.get("avg_price", raw_position.get("entry_price"))
237:    current_price = None
238:    if isinstance(latest_log, dict):
239:        current_price = latest_log.get("cert_bid", latest_log.get("cert_ask"))
240:    if current_price is None:
241:        current_price = raw_position.get("current_price")
242:    pnl_pct = None
243:    if _is_number(entry_price) and entry_price > 0 and _is_number(current_price):
244:        pnl_pct = ((current_price - entry_price) / entry_price) * 100.0
245:
246:    position = dict(raw_position)
247:    position["shares"] = quantity
248:    position["quantity"] = quantity
249:    position["side"] = raw_position.get("side") or raw_position.get("action") or "BUY"
250:    position["entry_price"] = entry_price
251:    position["avg_price"] = entry_price
252:    position["stop_price"] = raw_position.get("stop_price", raw_position.get("stop"))
253:    position["tp_price"] = raw_position.get("tp_price", raw_position.get("take_profit_price"))
254:    position["take_profit_price"] = position["tp_price"]
255:    position["current_price"] = current_price
256:    position["pnl_pct"] = _round_or_none(
257:        raw_position.get("pnl_pct") if raw_position.get("pnl_pct") is not None else pnl_pct,
258:        2,
259:    )
260:
261:    has_position = any(
262:        _is_number(value)
263:        for value in (quantity, entry_price, current_price)
264:    )
265:    return position if has_position else None
266:
267:
268:def _normalize_golddigger_log_entry(entry):
269:    if not isinstance(entry, dict):
270:        return None
271:    normalized = dict(entry)
272:    normalized.setdefault("composite_score", entry.get("S"))
273:    normalized.setdefault("z_gold", entry.get("z_g"))
274:    normalized.setdefault("z_fx", entry.get("z_f"))
275:    normalized.setdefault("z_yield", entry.get("z_y"))
276:    return normalized
277:
278:
279:def _normalize_golddigger_trade_entry(entry):
280:    if not isinstance(entry, dict):
281:        return None
282:
283:    quantity = entry.get("shares", entry.get("quantity"))
284:    price_sek = entry.get("price_sek")
285:    total_sek = entry.get("total_sek")
286:    if total_sek is None and _is_number(quantity) and _is_number(price_sek):
287:        total_sek = quantity * price_sek
288:
289:    normalized = dict(entry)
290:    normalized.setdefault("shares", quantity)
291:    normalized.setdefault("total_sek", _round_or_none(total_sek, 2))
292:    normalized.setdefault("composite_score", entry.get("composite_s"))
293:    return normalized
294:
295:
296:def _normalize_golddigger_state(state, log_entries):
297:    if not isinstance(state, dict) and not log_entries:
298:        return None
299:    state = dict(state or {})
300:    latest_log = log_entries[0] if log_entries else {}
301:    cfg = (_get_config() or {}).get("golddigger", {})
302:
303:    state["composite_score"] = latest_log.get("S", state.get("composite_score"))
304:    state["z_gold"] = latest_log.get("z_g", state.get("z_gold"))
305:    state["z_fx"] = latest_log.get("z_f", state.get("z_fx"))
306:    state["z_yield"] = latest_log.get("z_y", state.get("z_yield"))
307:    state["gold_price"] = latest_log.get("gold", state.get("gold_price"))
308:    state["usdsek"] = latest_log.get("usdsek", state.get("usdsek"))
309:    state["ts"] = latest_log.get("ts", state.get("last_poll_time"))
310:    state["theta_in"] = cfg.get("theta_in", state.get("theta_in", 0.7))
311:    state["theta_out"] = cfg.get("theta_out", state.get("theta_out", 0.1))
312:    state["spread_max"] = cfg.get("spread_max", state.get("spread_max", 0.02))
313:    state["confirm_required"] = cfg.get("confirm_polls", state.get("confirm_required", 3))
314:    state["risk_fraction"] = cfg.get("risk_fraction", state.get("risk_fraction", 0.005))
315:    state["max_notional_fraction"] = cfg.get("max_notional_fraction", state.get("max_notional_fraction", 0.10))
316:    state["leverage"] = cfg.get("leverage", state.get("leverage", 20.0))
317:
318:    if state.get("confirm_count") is None:
319:        state["confirm_count"] = 0
320:    if log_entries:
321:        confirms = 0
322:        theta_in = state.get("theta_in", 0.7)
323:        for entry in log_entries:
324:            score = entry.get("S")
325:            z_gold = entry.get("z_g")
326:            if not _is_number(score) or score < theta_in or (z_gold is not None and z_gold <= 0):
327:                break
328:            confirms += 1
329:        state["confirm_count"] = confirms
330:
331:    latest_ts = _parse_iso8601(state.get("last_poll_time") or latest_log.get("ts"))
332:    max_age_seconds = max(60, int(cfg.get("poll_seconds", 5)) * 12)
333:    state["session_active"] = (
334:        latest_ts is not None
335:        and not bool(state.get("halted"))
336:        and (datetime.now(UTC) - latest_ts).total_seconds() <= max_age_seconds
337:    )
338:    state["daily"] = {
339:        "trade_count": state.get("daily_trades", 0),
340:        "max_trades": cfg.get("max_daily_trades", 10),
341:        "pnl_sek": state.get("daily_pnl"),
342:    }
343:    state["position"] = _normalize_golddigger_position(state.get("position"), latest_log)
344:    return state
345:
346:
347:def _normalize_metals_llm_predictions(raw_llm):
348:    if not isinstance(raw_llm, dict):
349:        return {}
350:
351:    predictions = {}
352:    for ticker, payload in raw_llm.items():
353:        if not isinstance(payload, dict):
354:            continue
355:
356:        consensus_action = payload.get("consensus_action") or payload.get("consensus")
357:        consensus_direction = payload.get("consensus_dir")
358:        pred = {
359:            "consensus": {
360:                "weighted_action": consensus_action,
361:                "direction": consensus_direction,
362:                "confidence": payload.get("consensus_conf"),
363:            }
364:        }
365:
366:        ministral_action = payload.get("ministral")
367:        if ministral_action:
368:            pred["ministral"] = {
369:                "action": ministral_action,
370:                "confidence": payload.get("ministral_conf"),
371:            }
372:
373:        for horizon in ("1h", "3h"):
374:            prefix = f"chronos_{horizon}"
375:            direction = payload.get(prefix)
376:            pct_move = payload.get(f"{prefix}_pct_move")
377:            if pct_move is None:
378:                pct_move = payload.get(f"{prefix}_pct")
379:            confidence = payload.get(f"{prefix}_conf")
380:            if direction is None and pct_move is None and confidence is None:
381:                continue
382:            pred[prefix] = {
383:                "direction": direction,
384:                "pct_move": pct_move,
385:                "confidence": confidence,
386:            }
387:
388:        predictions[ticker] = pred
389:
390:    return {
391:        "age_seconds": None,
392:        "models": ["ministral", "chronos_1h", "chronos_3h"],
393:        "accuracy": {},
394:        "predictions": predictions,
395:    }
396:
397:
398:def _normalize_metals_forecast_signals(raw_llm):
399:    if not isinstance(raw_llm, dict):
400:        return {}
401:
402:    signals = {}
403:    for ticker, payload in raw_llm.items():
404:        if not isinstance(payload, dict):
405:            continue
406:        chronos_1h_pct = payload.get("chronos_1h_pct_move")
407:        if chronos_1h_pct is None:
408:            chronos_1h_pct = payload.get("chronos_1h_pct")
409:        action = payload.get("consensus_action") or payload.get("consensus")
410:        if action is None and chronos_1h_pct is None:
411:            continue
412:        signals[ticker] = {
413:            "action": action,
414:            "chronos_1h_pct": chronos_1h_pct,
415:        }
416:
417:    return {"forecast_signals": signals} if signals else {}
418:
419:
420:def _normalize_metals_decisions(decisions):
421:    normalized = []
422:    for entry in decisions:
423:        if not isinstance(entry, dict):
424:            continue
425:        item = dict(entry)
426:        action = (item.get("action") or (item.get("prediction") or {}).get("action") or "HOLD").upper()
427:        positions = {}
428:        for key, payload in (item.get("positions") or {}).items():
429:            if not isinstance(payload, dict):
430:                continue
431:            pos_item = dict(payload)
432:            pos_item.setdefault("action", action)
433:            positions[key] = pos_item
434:        item["positions"] = positions
435:        if not item.get("reasoning"):
436:            item["reasoning"] = item.get("trigger") or item.get("thesis_status") or ""
437:        normalized.append(item)
438:    return normalized
439:
440:
441:def _drawdown_level_from_pct(drawdown_pct):
442:    if not _is_number(drawdown_pct):
443:        return "UNKNOWN"
444:    if drawdown_pct <= -15.0:
445:        return "EMERGENCY"
446:    if drawdown_pct <= -10.0:
447:        return "WARNING"
448:    return "OK"
449:
450:
451:def _normalize_metals_risk(risk):
452:    if not isinstance(risk, dict):
453:        return {}
454:
455:    item = dict(risk)
456:    drawdown = item.get("drawdown")
457:    if isinstance(drawdown, dict) and "level" not in drawdown:
458:        drawdown = dict(drawdown)
459:        drawdown["level"] = _drawdown_level_from_pct(drawdown.get("current_drawdown_pct"))
460:        item["drawdown"] = drawdown
461:
462:    trade_guards = item.get("trade_guards")
463:    if isinstance(trade_guards, dict) and "status" not in trade_guards:
464:        tg_item = dict(trade_guards)
465:        tg_item["status"] = "warnings" if tg_item else "unknown"
466:        item["trade_guards"] = tg_item
467:
468:    return item
469:
470:
471:def _normalize_metals_context(context):
472:    if not isinstance(context, dict):
473:        return context
474:    item = dict(context)
475:    item["risk"] = _normalize_metals_risk(item.get("risk"))
476:    return item
477:
478:
479:def _merge_missing_structure(primary, fallback):
480:    if primary is None:
481:        return fallback
482:    if fallback is None:
483:        return primary
484:    if isinstance(primary, dict) and isinstance(fallback, dict):
485:        merged = dict(primary)
486:        for key, fallback_value in fallback.items():
487:            primary_value = merged.get(key)
488:            if primary_value is None:
489:                merged[key] = fallback_value
490:                continue
491:            if isinstance(primary_value, dict) and not primary_value:
492:                merged[key] = fallback_value
493:                continue
494:            if isinstance(primary_value, list) and not primary_value:
495:                merged[key] = fallback_value
496:                continue
497:            merged[key] = _merge_missing_structure(primary_value, fallback_value)
498:        return merged
499:    return primary
500:
501:
502:def _build_metals_context_fallback(decisions):
503:    positions_state = _read_json(DATA_DIR / "metals_positions_state.json") or {}
504:    signal_entries = list(_iter_latest_dict_entries(DATA_DIR / "metals_signal_log.jsonl", read_limit=10))
505:    value_history = _read_jsonl(DATA_DIR / "metals_value_history.jsonl", limit=10)
506:    technicals = _read_json(DATA_DIR / "silver_analysis.json") or {}
507:    latest_signal = signal_entries[0] if signal_entries else {}
508:    latest_value = value_history[-1] if value_history else {}
509:    latest_decision = decisions[0] if decisions else {}
510:
511:    if not positions_state and not latest_signal and not latest_value and not latest_decision:
512:        return None
513:
514:    prices = latest_signal.get("prices", {}) if isinstance(latest_signal, dict) else {}
515:    latest_decision_positions = latest_decision.get("positions", {}) if isinstance(latest_decision, dict) else {}
516:    latest_value_positions = latest_value.get("positions", {}) if isinstance(latest_value, dict) else {}
517:
518:    active_keys = [
519:        key for key, payload in positions_state.items()
520:        if isinstance(payload, dict) and payload.get("active")
521:    ]
522:    if not active_keys:
523:        active_keys = list(latest_decision_positions.keys())
524:
525:    positions = {}
526:    total_invested = latest_value.get("total_invested")
527:    total_value = latest_value.get("total_value")
528:    total_pnl_pct = latest_value.get("pnl_pct")
529:
530:    for key in active_keys:
531:        state_payload = positions_state.get(key, {})
532:        decision_payload = latest_decision_positions.get(key, {})
533:        value_payload = latest_value_positions.get(key, {})
534:
535:        units = state_payload.get("units", decision_payload.get("units"))
536:        entry = state_payload.get("entry", decision_payload.get("entry"))
537:        stop = state_payload.get("stop", decision_payload.get("stop"))
538:        bid = prices.get(key, value_payload.get("bid", decision_payload.get("bid")))
539:        invested = (units * entry) if _is_number(units) and _is_number(entry) else None
540:        value_sek = value_payload.get("value")
541:        if value_sek is None and _is_number(units) and _is_number(bid):
542:            value_sek = units * bid
543:        profit_sek = None
544:        if _is_number(value_sek) and _is_number(invested):
545:            profit_sek = value_sek - invested
546:
547:        pnl_pct = value_payload.get("pnl_pct", decision_payload.get("pnl_pct"))
548:        if pnl_pct is None and _is_number(entry) and entry > 0 and _is_number(bid):
549:            pnl_pct = ((bid - entry) / entry) * 100.0
550:
551:        dist_stop_pct = decision_payload.get("dist_stop_pct")
552:        if dist_stop_pct is None and _is_number(bid) and bid > 0 and _is_number(stop):
553:            dist_stop_pct = ((bid - stop) / bid) * 100.0
554:
555:        positions[key] = {
556:            "name": decision_payload.get("name", key),
557:            "units": units,
558:            "entry": entry,
559:            "bid": bid,
560:            "ask": prices.get(f"{key}_ask"),
561:            "pnl_pct": _round_or_none(pnl_pct, 2),
562:            "value_sek": _round_or_none(value_sek, 1),
563:            "invested_sek": _round_or_none(invested, 1),
564:            "profit_sek": _round_or_none(profit_sek, 1),
565:            "peak_bid": None,
566:            "from_peak_pct": _round_or_none(decision_payload.get("from_peak_pct"), 2),
567:            "stop": stop,
568:            "dist_to_stop_pct": _round_or_none(dist_stop_pct, 2),
569:            "day_change_pct": None,
570:            "leverage": None,
571:            "barrier": None,
572:            "active": True,
573:        }
574:
575:    if total_invested is None:
576:        invested_values = [payload.get("invested_sek") for payload in positions.values() if _is_number(payload.get("invested_sek"))]
577:        total_invested = sum(invested_values) if invested_values else None
578:    if total_value is None:
579:        value_values = [payload.get("value_sek") for payload in positions.values() if _is_number(payload.get("value_sek"))]
580:        total_value = sum(value_values) if value_values else None
581:    if total_pnl_pct is None and _is_number(total_invested) and total_invested > 0 and _is_number(total_value):
582:        total_pnl_pct = ((total_value / total_invested) - 1.0) * 100.0
583:
584:    drawdown_pct = None
585:    if isinstance(latest_decision, dict):
586:        drawdown_pct = (latest_decision.get("risk") or {}).get("drawdown_pct")
587:
588:    price_history_recent = []
589:    gold_fallback = ((technicals.get("context") or {}).get("gold_price"))
590:    for entry in reversed(signal_entries):
591:        snap_prices = entry.get("prices", {})
592:        price_history_recent.append({
593:            "ts": entry.get("ts"),
594:            "gold": snap_prices.get("gold") or snap_prices.get("XAU-USD") or gold_fallback,
595:            "gold_und": snap_prices.get("gold_und") or snap_prices.get("XAU-USD"),
596:            "silver79": snap_prices.get("silver79"),
597:            "silver79_und": snap_prices.get("silver79_und") or snap_prices.get("XAG-USD"),
598:            "silver301": snap_prices.get("silver301"),
599:            "silver301_und": snap_prices.get("silver301_und") or snap_prices.get("XAG-USD"),
600:        })
601:
602:    silver_price = (
603:        prices.get("XAG-USD")
604:        or prices.get("silver301_und")
605:        or prices.get("silver79_und")
606:        or ((technicals.get("price") or {}).get("current"))
607:    )
608:    gold_price = prices.get("XAU-USD") or prices.get("gold_und") or gold_fallback
609:    now_sthlm = _stockholm_now()
610:
611:    return {
612:        "timestamp": latest_signal.get("ts") or latest_decision.get("ts"),
613:        "cet_time": now_sthlm.strftime("%H:%M %Z"),
614:        "check_count": latest_signal.get("check") or latest_decision.get("check_count"),
615:        "invoke_count": latest_decision.get("invoke_count"),
616:        "trigger_reason": (
617:            (latest_signal.get("trigger_reasons") or [None])[0]
618:            or latest_decision.get("trigger")
619:        ),
620:        "tier": latest_decision.get("tier"),
621:        "market_close_cet": "21:30",
622:        "hours_remaining": _hours_until_stockholm_close(now_sthlm),
623:        "positions": positions,
624:        "underlying": {
625:            "gold": {"price": gold_price} if gold_price is not None else {},
626:            "silver": {"price": silver_price} if silver_price is not None else {},
627:        },
628:        "totals": {
629:            "invested": _round_or_none(total_invested, 0),
630:            "current": _round_or_none(total_value, 0),
631:            "pnl_pct": _round_or_none(total_pnl_pct, 2),
632:            "profit_sek": _round_or_none(
633:                (total_value - total_invested)
634:                if _is_number(total_value) and _is_number(total_invested)
635:                else None,
636:                0,
637:            ),
638:        },
639:        "price_history_recent": price_history_recent,
640:        "signals": _merge_missing_structure(
641:            latest_signal.get("signals", {}),
642:            _normalize_metals_forecast_signals(
643:                latest_signal.get("llm") or latest_decision.get("llm")
644:            ),
645:        ),
646:        "recent_decisions": decisions[:5],
647:        "short_instruments": {},
648:        "llm_predictions": _normalize_metals_llm_predictions(
649:            latest_signal.get("llm") or latest_decision.get("llm")
650:        ),
651:        "risk": {
652:            "monte_carlo": {},
653:            "drawdown": {
654:                "current_drawdown_pct": _round_or_none(drawdown_pct, 2),
655:                "level": _drawdown_level_from_pct(drawdown_pct),
656:            },
657:            "trade_guards": {
658:                "status": "warnings" if latest_signal.get("triggered") else "all_clear",
659:                "reason": "; ".join((latest_signal.get("trigger_reasons") or [])[:2]) or None,
660:            },
661:        },
662:        "trades_today_file": "data/metals_trades.jsonl",
663:    }
664:
665:
666:def _aggregate_accuracy_bucket(bucket):
667:    """Aggregate nested accuracy stats into one accuracy/total pair."""
668:    if not isinstance(bucket, dict):
669:        return {"accuracy": None, "total": 0, "correct": 0}
670:
671:    correct = 0
672:    total = 0
673:    for stats in bucket.values():
674:        if not isinstance(stats, dict):
675:            continue
676:        correct += int(stats.get("correct", 0) or 0)
677:        total += int(stats.get("total", 0) or 0)
678:
679:    return {
680:        "accuracy": round(correct / total, 3) if total else None,
681:        "correct": correct,
682:        "total": total,
683:    }
684:
685:
686:def _build_local_llm_trend_point(entry, ticker=None):
687:    """Flatten one local-LLM history entry into chart-friendly metrics."""
688:    ticker = (ticker or "").upper() or None
689:    ministral = ((entry.get("ministral") or {}).get("overall") or {})
690:    by_ticker = ((entry.get("ministral") or {}).get("by_ticker") or {})
691:    ticker_stats = by_ticker.get(ticker, {}) if ticker else {}
692:    health = entry.get("health") or {}
693:    forecast = entry.get("forecast") or {}
694:    gating = (entry.get("gating_counts") or {}).get("forecast") or {}
695:
696:    raw_1h = _aggregate_accuracy_bucket((forecast.get("raw") or {}).get("1h"))
697:    raw_24h = _aggregate_accuracy_bucket((forecast.get("raw") or {}).get("24h"))
698:    effective_1h = _aggregate_accuracy_bucket((forecast.get("effective") or {}).get("1h"))
699:    effective_24h = _aggregate_accuracy_bucket((forecast.get("effective") or {}).get("24h"))
700:
701:    return {
702:        "date": entry.get("date"),
703:        "exported_at": entry.get("exported_at"),
704:        "days": entry.get("days"),
705:        "ticker": ticker,
706:        "ministral_accuracy": ministral.get("accuracy"),
707:        "ministral_samples": ministral.get("samples", 0),
708:        "ministral_ticker_accuracy": ticker_stats.get("accuracy"),
709:        "ministral_ticker_samples": ticker_stats.get("samples", 0),
710:        "chronos_success_rate": (health.get("chronos") or {}).get("success_rate"),
711:        "chronos_total": (health.get("chronos") or {}).get("total", 0),
712:        "kronos_success_rate": (health.get("kronos") or {}).get("success_rate"),
713:        "kronos_total": (health.get("kronos") or {}).get("total", 0),
714:        "forecast_raw_1h_accuracy": raw_1h["accuracy"],
715:        "forecast_raw_1h_total": raw_1h["total"],
716:        "forecast_raw_24h_accuracy": raw_24h["accuracy"],
717:        "forecast_raw_24h_total": raw_24h["total"],
718:        "forecast_effective_1h_accuracy": effective_1h["accuracy"],
719:        "forecast_effective_1h_total": effective_1h["total"],
720:        "forecast_effective_24h_accuracy": effective_24h["accuracy"],
721:        "forecast_effective_24h_total": effective_24h["total"],
722:        "forecast_gating_raw": gating.get("raw", 0),
723:        "forecast_gating_held": gating.get("held", 0),
724:        "forecast_gating_insufficient_data": gating.get("insufficient_data", 0),
725:        "forecast_gating_vol_gated": gating.get("vol_gated", 0),
726:    }
727:
728:
729:# ---------------------------------------------------------------------------
730:# Token authentication middleware
731:# ---------------------------------------------------------------------------
732:
733:# Auth + cookie machinery moved to dashboard/auth.py on 2026-05-02 to break
734:# the circular import with dashboard/house_blueprint.py. We re-import here
735:# so existing references (`require_auth`, `COOKIE_NAME`, etc.) keep working
736:# inside this module's body, and so any lingering external code that does
737:# `from dashboard.app import require_auth` still resolves. Tests should
738:# patch `dashboard.auth.*` directly — patches on `dashboard.app.*` will not
739:# take effect since require_auth resolves names via dashboard.auth's
740:# module globals.
741:from dashboard.auth import (  # noqa: E402
742:    COOKIE_MAX_AGE,
743:    COOKIE_NAME,
744:    _get_config as _auth_get_config,  # noqa: F401 — kept for compat
745:    _get_dashboard_token,
746:    _refresh_cookie,
747:    require_auth,
748:)
749:
750:
751:# ---------------------------------------------------------------------------
752:# Routes — Static
753:# ---------------------------------------------------------------------------
754:
755:@app.route("/")
756:@require_auth
757:def index():
758:    # If the user arrived via ?token=XXX, the cookie was just set in
759:    # require_auth. Redirect to a token-less URL so the address bar (and
760:    # whatever the user bookmarks next) stays clean. The redirect inherits
761:    # the Set-Cookie from require_auth's wrapped response.
762:    if request.args.get("token"):
763:        return redirect("/", code=302)
764:    return send_from_directory("static", "index.html")
765:
766:
767:@app.route("/legacy")
768:@require_auth
769:def index_legacy():
770:    # Pre-redesign single-file dashboard preserved as a fallback during the
771:    # 2026-05-03 mobile-first rollout. See docs/PLAN.md.
772:    if request.args.get("token"):
773:        return redirect("/legacy", code=302)
774:    return send_from_directory("static", "index_legacy.html")
775:
776:
777:@app.route("/logout")
778:def logout():
779:    """Clear the pf_dashboard_token cookie and redirect to /.
780:
781:    The auth cookie is HttpOnly, so client JS cannot expire it via
782:    document.cookie — the browser ignores any attempt to write a name that
783:    Set-Cookie marked HttpOnly. The mobile Settings → Sign out button
784:    therefore has to navigate here so the server can emit the matching
785:    Set-Cookie with Max-Age=0. (Codex P2 finding 2026-05-03.)
786:
787:    No `require_auth`: an unauthenticated visitor hitting /logout still gets
788:    the cookie wiped (harmless — they had no valid cookie anyway) and
789:    Cloudflare Access still gates the redirected destination.
790:    """
791:    response = redirect("/", code=302)
792:    # Match every flag we set on the original cookie except expiry.
793:    response.set_cookie(
794:        "pf_dashboard_token",
795:        "",
796:        max_age=0,
797:        expires=0,
798:        httponly=True,
799:        secure=True,
800:        samesite="Lax",
801:    )
802:    return response
803:
804:
805:# ---------------------------------------------------------------------------
806:# Routes — API (all require auth)
807:# ---------------------------------------------------------------------------
808:
809:@app.route("/api/summary")
810:@require_auth
811:def api_summary():
812:    """Combined endpoint for auto-refresh: signals + both portfolios + telegrams."""
813:    sig = _read_json(DATA_DIR / "agent_summary.json")
814:    port = _read_json(DATA_DIR / "portfolio_state.json")
815:    port_bold = _read_json(DATA_DIR / "portfolio_state_bold.json")
816:    tel = list(_iter_latest_dict_entries(DATA_DIR / "telegram_messages.jsonl", read_limit=50))
817:    return jsonify({
818:        "signals": sig,
819:        "portfolio": port,
820:        "portfolio_bold": port_bold,
821:        "telegrams": tel,
822:    })
823:
824:
825:@app.route("/api/signals")
826:@require_auth
827:def api_signals():
828:    data = _read_json(DATA_DIR / "agent_summary.json")
829:    if not data:
830:        return jsonify({"error": "no data"}), 404
831:    return jsonify(data)
832:
833:
834:@app.route("/api/portfolio")
835:@require_auth
836:def api_portfolio():
837:    data = _read_json(DATA_DIR / "portfolio_state.json")
838:    if not data:
839:        return jsonify({"error": "no data"}), 404
840:    return jsonify(data)
841:
842:
843:@app.route("/api/portfolio-bold")
844:@require_auth
845:def api_portfolio_bold():
846:    data = _read_json(DATA_DIR / "portfolio_state_bold.json")
847:    if not data:
848:        return jsonify({"error": "no data"}), 404
849:    return jsonify(data)
850:
851:
852:@app.route("/api/grid-fisher")
853:@require_auth
854:def api_grid_fisher():
855:    """Grid market-maker state + recent decisions.
856:
857:    Returns:
858:        {
859:          "state": <data/grid_fisher_state.json>,
860:          "recent_decisions": [last 50 entries from grid_fisher_decisions.jsonl]
861:        }
862:    """
863:    state = _read_json(DATA_DIR / "grid_fisher_state.json") or {}
864:    decisions: list[dict] = []
865:    decisions_path = DATA_DIR / "grid_fisher_decisions.jsonl"
866:    try:
867:        if decisions_path.exists():
868:            from portfolio.file_utils import load_jsonl_tail
869:            decisions = load_jsonl_tail(decisions_path, max_entries=50)
870:    except Exception:
871:        decisions = []
872:    return jsonify({"state": state, "recent_decisions": decisions})
873:
874:
875:@app.route("/api/mstr_loop")
876:@require_auth
877:def api_mstr_loop():
878:    """Live snapshot of the MSTR Loop bot (v2 Tier 3).
879:
880:    Returns state + scorecard + latest poll in one JSON:
881:        {
882:          "state": {cash, positions, total_pnl, ...},
883:          "scorecard": {win_rate, expectancy, trades_by_strategy, ...},
884:          "last_poll": {last cycle snapshot from mstr_loop_poll.jsonl},
885:          "last_trade": {last closed trade from mstr_loop_trades.jsonl},
886:        }
887:    """
888:    out = {
889:        "state": _read_json(DATA_DIR / "mstr_loop_state.json") or {},
890:        "scorecard": _read_json(DATA_DIR / "mstr_loop_scorecard.json") or {},
891:        "last_poll": None,
892:        "last_trade": None,
893:    }
894:    import json as _json
895:    poll_path = DATA_DIR / "mstr_loop_poll.jsonl"
896:    if poll_path.exists():
897:        try:
898:            with open(poll_path, encoding="utf-8") as f:
899:                for line in f:
900:                    if line.strip():
901:                        try:
902:                            out["last_poll"] = _json.loads(line)
903:                        except _json.JSONDecodeError:
904:                            pass
905:        except OSError:
906:            pass
907:    trades_path = DATA_DIR / "mstr_loop_trades.jsonl"
908:    if trades_path.exists():
909:        try:
910:            with open(trades_path, encoding="utf-8") as f:
911:                for line in f:
912:                    if line.strip():
913:                        try:
914:                            out["last_trade"] = _json.loads(line)
915:                        except _json.JSONDecodeError:
916:                            pass
917:        except OSError:
918:            pass
919:    return jsonify(out)
920:
921:
922:@app.route("/api/invocations")
923:@require_auth
924:def api_invocations():
925:    entries = _read_jsonl(DATA_DIR / "invocations.jsonl", limit=50)
926:    return jsonify(entries)
927:
928:
929:@app.route("/api/telegrams")
930:@require_auth
931:def api_telegrams():
932:    """Return telegram messages with optional filtering.
933:
934:    Query params:
935:      - limit: max entries (default 200, max 2000)
936:      - category: filter by category (trade, analysis, iskbets, bigbet, digest, etc.)
937:      - search: text search in message body
938:    """
939:    limit = _parse_limit_arg("limit", default=200, max_value=2000)
940:    category_filter = request.args.get("category", "").strip().lower()
941:    search_filter = request.args.get("search", "").strip().lower()
942:
943:    results = []
944:    for entry in _iter_latest_dict_entries(DATA_DIR / "telegram_messages.jsonl", read_limit=5000):
945:        if category_filter and (entry.get("category", "") or "").lower() != category_filter:
946:            continue
947:        if search_filter and search_filter not in (entry.get("text", "") or "").lower():
948:            continue
949:        results.append(entry)
950:        if len(results) >= limit:
951:            break
952:
953:    return jsonify(results)
954:
955:
956:@app.route("/api/signal-log")
957:@require_auth
958:def api_signal_log():
959:    entries = _read_jsonl(DATA_DIR / "signal_log.jsonl", limit=50)
960:    return jsonify(entries)
961:
962:
963:_API_ACCURACY_CACHE: dict = {"ts": 0.0, "data": None}
964:_API_ACCURACY_TTL_SEC = 60.0
965:
966:
967:@app.route("/api/accuracy")
968:@require_auth
969:def api_accuracy():
970:    """Aggregate accuracy report across 4 horizons.
971:
972:    2026-05-03: previously took >15s (timed out from clients) because
973:    each request did 12 full signal-log scans (4 horizons × 3 metrics).
974:    Now backed by accuracy_stats.get_or_compute_*() which read
975:    accuracy_cache.json on the hot path, plus a 60s in-process TTL
976:    that coalesces burst requests during dashboard polling.
977:    """
978:    import time
979:    now = time.time()
980:    if (_API_ACCURACY_CACHE["data"] is not None
981:            and (now - _API_ACCURACY_CACHE["ts"]) < _API_ACCURACY_TTL_SEC):
982:        return jsonify(_API_ACCURACY_CACHE["data"])
983:
984:    try:
985:        from portfolio.accuracy_stats import (
986:            get_or_compute_accuracy,
987:            get_or_compute_consensus_accuracy,
988:            get_or_compute_per_ticker_accuracy,
989:        )
990:        from portfolio.tickers import DISABLED_SIGNALS, get_disabled_reason
991:
992:        def _enrich_signals(signals_dict):
993:            # 2026-05-05: enrich at response time so older cached entries
994:            # (written before signal_accuracy() learned to emit `samples`/
995:            # `enabled`) still render correctly on the dashboard. The
996:            # accuracy cache has a 1h TTL; without this fallback the
997:            # disabled-signal labels would not appear until the cache
998:            # rebuilds.
999:            #
1000:            # Important: `enabled` and `disabled_reason` are *overwritten*
1001:            # from the live DISABLED_SIGNALS, not setdefault'd. A signal
1002:            # re-enabled (e.g. statistical_jump_regime, 2026-04-29) or
1003:            # newly disabled would otherwise keep the stale flag from the
1004:            # cache file until the next 1h rebuild. `samples` is just an
1005:            # alias for `total` so setdefault is fine there.
1006:            if not isinstance(signals_dict, dict):
1007:                return signals_dict
1008:            for sig_name, info in signals_dict.items():
1009:                if not isinstance(info, dict):
1010:                    continue
1011:                if "samples" not in info and "total" in info:
1012:                    info["samples"] = info["total"]
1013:                enabled = sig_name not in DISABLED_SIGNALS
1014:                info["enabled"] = enabled
1015:                if enabled:
1016:                    info.pop("disabled_reason", None)
1017:                else:
1018:                    reason = get_disabled_reason(sig_name)
1019:                    if reason:
1020:                        info["disabled_reason"] = reason
1021:                    else:
1022:                        info.pop("disabled_reason", None)
1023:            return signals_dict
1024:
1025:        result = {}
1026:        for horizon in ["1d", "3d", "5d", "10d"]:
1027:            sa = get_or_compute_accuracy(horizon)
1028:            ca = get_or_compute_consensus_accuracy(horizon)
1029:            ta = get_or_compute_per_ticker_accuracy(horizon)
1030:            # ca/sa/ta may be None when the underlying cache miss returned
1031:            # no data (cold cache + no signal-log entries yet); skip those
1032:            # horizons entirely so the response stays well-formed.
1033:            if ca and ca.get("total", 0) > 0:
1034:                result[horizon] = {
1035:                    "signals": _enrich_signals(sa or {}),
1036:                    "consensus": ca,
1037:                    "per_ticker": ta or {},
1038:                }
1039:        _API_ACCURACY_CACHE["data"] = result
1040:        _API_ACCURACY_CACHE["ts"] = now
1041:        return jsonify(result)
1042:    except Exception:
1043:        logger.exception("accuracy endpoint error")
1044:        return jsonify({"error": "Internal server error"}), 500
1045:
1046:
1047:@app.route("/api/iskbets")
1048:@require_auth
1049:def api_iskbets():
1050:    config = _read_json(DATA_DIR / "iskbets_config.json")
1051:    state = _read_json(DATA_DIR / "iskbets_state.json")
1052:    return jsonify({"config": config, "state": state})
1053:
1054:
1055:@app.route("/api/lora-status")
1056:@require_auth
1057:def api_lora_status():
1058:    state = _read_json(TRAINING_DIR / "state.json")
1059:    progress = _read_json(TRAINING_DIR / "training_progress.json")
1060:    return jsonify({"state": state, "training_progress": progress})
1061:
1062:
1063:# ---------------------------------------------------------------------------
1064:# New: Portfolio validation
1065:# ---------------------------------------------------------------------------
1066:
1067:@app.route("/api/validate-portfolio", methods=["POST"])
1068:@require_auth
1069:def api_validate_portfolio():
1070:    """Validate a portfolio JSON for integrity.
1071:
1072:    Delegates to portfolio_validator.validate_portfolio() which performs
1073:    comprehensive checks: cash, holdings, fees, transactions, avg_cost.
1074:    """
1075:    data = request.get_json(silent=True)
1076:    if not data:
1077:        return jsonify({"valid": False, "errors": ["No JSON body provided"]}), 400
1078:
1079:    try:
1080:        from portfolio.portfolio_validator import validate_portfolio
1081:        errors = validate_portfolio(data)
1082:    except Exception as e:
1083:        return jsonify({"valid": False, "errors": [f"Validation error: {e}"]}), 500
1084:
1085:    return jsonify({
1086:        "valid": len(errors) == 0,
1087:        "errors": errors,
1088:    })
1089:
1090:
1091:# ---------------------------------------------------------------------------
1092:# New: Equity curve
1093:# ---------------------------------------------------------------------------
1094:
1095:@app.route("/api/equity-curve")
1096:@require_auth
1097:def api_equity_curve():
1098:    """Return portfolio value history for charting.
1099:
1100:    Reads data/portfolio_value_history.jsonl. Returns empty array if missing.
1101:    """
1102:    entries = _read_jsonl(DATA_DIR / "portfolio_value_history.jsonl", limit=5000)
1103:    return jsonify(entries)
1104:
1105:
1106:# ---------------------------------------------------------------------------
1107:# New: Signal heatmap (30 signals x all tickers)
1108:# ---------------------------------------------------------------------------
1109:
1110:@app.route("/api/signal-heatmap")
1111:@require_auth
1112:def api_signal_heatmap():
1113:    """Return the full 30-signal x all-tickers grid.
1114:
1115:    Each cell is BUY/SELL/HOLD. Built from agent_summary.json signals + enhanced_signals.
1116:    """
1117:    summary = _read_json(DATA_DIR / "agent_summary.json")
1118:    if not summary:
1119:        return jsonify({"error": "no data"}), 404
1120:
1121:    signals_data = summary.get("signals", {})
1122:
1123:    # Core signal names (11 total: 8 active + 3 disabled)
1124:    core_signals = [
1125:        "rsi", "macd", "ema", "bb", "fear_greed", "sentiment",
1126:        "ministral", "volume", "ml", "funding", "custom_lora"
1127:    ]
1128:    # Enhanced composite signal names (19 modules, signals #12-#30)
1129:    enhanced_signals = [
1130:        "trend", "momentum", "volume_flow", "volatility_sig",
1131:        "candlestick", "structure", "fibonacci", "smart_money",
1132:        "oscillators", "heikin_ashi", "mean_reversion", "calendar",
1133:        "macro_regime", "momentum_factors", "news_event", "econ_calendar",
1134:        "forecast", "claude_fundamental", "futures_flow"
1135:    ]
1136:    all_signals = core_signals + enhanced_signals
1137:
1138:    heatmap = {}
1139:    tickers = list(signals_data.keys())
1140:
1141:    for ticker in tickers:
1142:        sig = signals_data[ticker]
1143:        extra = sig.get("extra", {})
1144:        votes = extra.get("_votes", {})
1145:
1146:        # _votes contains all 30 signal keys (core + enhanced)
1147:        row = {}
1148:        for s in all_signals:
1149:            row[s] = (votes.get(s, "HOLD") or "HOLD").upper()
1150:        heatmap[ticker] = row
1151:
1152:    # Per-(ticker, signal) state-change timestamps for the "time-in-state" badge.
1153:    # Written by portfolio.reporting._update_signal_state_since each loop cycle.
1154:    # Missing or malformed payload degrades to an empty map: frontend renders
1155:    # cells without the badge — never 500.
1156:    #
1157:    # Codex P2 (2026-05-05): the since-file is written *before* agent_summary
1158:    # in the same cycle, and a swallowed write-failure can also leave the two
1159:    # out of sync. Guard against showing a stale duration on a freshly-flipped
1160:    # vote by only emitting `since` when the recorded vote matches the current
1161:    # heatmap value. Mismatched cells fall back to colour-only until the next
1162:    # cycle re-syncs both files.
1163:    state_since_payload = _read_json(DATA_DIR / "signal_state_since.json") or {}
1164:    state_since_votes = state_since_payload.get("votes") if isinstance(state_since_payload, dict) else None
1165:    since: dict[str, dict[str, str]] = {}
1166:    if isinstance(state_since_votes, dict):
1167:        for ticker in tickers:
1168:            tk_state = state_since_votes.get(ticker)
1169:            if not isinstance(tk_state, dict):
1170:                continue
1171:            row_since: dict[str, str] = {}
1172:            current_row = heatmap.get(ticker, {})
1173:            for s in all_signals:
1174:                entry = tk_state.get(s)
1175:                if not isinstance(entry, dict):
1176:                    continue
1177:                since_ts = entry.get("since")
1178:                if not isinstance(since_ts, str):
1179:                    continue
1180:                if entry.get("vote") != current_row.get(s):
1181:                    continue  # stale: vote in since-file disagrees with heatmap
1182:                row_since[s] = since_ts
1183:            if row_since:
1184:                since[ticker] = row_since
1185:
1186:    # 2026-05-05: ship the disabled set so the heatmap can render
1187:    # disabled cells with the muted style + tap-to-show reason. The
1188:    # frontend already reads `data.disabled_signals` (signals.js:137).
1189:    try:
1190:        from portfolio.tickers import DISABLED_SIGNALS
1191:        disabled = sorted(DISABLED_SIGNALS)
1192:    except Exception:
1193:        disabled = []
1194:    return jsonify({
1195:        "tickers": tickers,
1196:        "signals": all_signals,
1197:        "core_signals": core_signals,
1198:        "enhanced_signals": enhanced_signals,
1199:        "heatmap": heatmap,
1200:        "since": since,
1201:        "disabled_signals": disabled,
1202:    })
1203:
1204:
1205:# ---------------------------------------------------------------------------
1206:# New: Trigger activity timeline
1207:# ---------------------------------------------------------------------------
1208:
1209:@app.route("/api/triggers")
1210:@require_auth
1211:def api_triggers():
1212:    """Return last 50 trigger/invocation events from invocations.jsonl."""
1213:    entries = _read_jsonl(DATA_DIR / "invocations.jsonl", limit=50)
1214:    return jsonify(entries)
1215:
1216:
1217:@app.route("/api/accuracy-history")
1218:@require_auth
1219:def api_accuracy_history():
1220:    """Return accuracy snapshots over time for charting trend lines.
1221:
1222:    2026-05-05: tag each per-signal slice with `enabled` so the chart
1223:    can dim/exclude force-HOLD'd signals. Tag is derived at response
1224:    time from DISABLED_SIGNALS so historical snapshots written before
1225:    the flag existed are also tagged correctly.
1226:    """
1227:    entries = _read_jsonl(DATA_DIR / "accuracy_snapshots.jsonl", limit=500)
1228:    try:
1229:        from portfolio.tickers import DISABLED_SIGNALS
1230:        for snap in entries:
1231:            sigs = snap.get("signals") if isinstance(snap, dict) else None
1232:            if not isinstance(sigs, dict):
1233:                continue
1234:            for sig_name, info in sigs.items():
1235:                if isinstance(info, dict):
1236:                    # Overwrite (not setdefault) — see /api/accuracy comment.
1237:                    info["enabled"] = sig_name not in DISABLED_SIGNALS
1238:    except Exception:
1239:        logger.exception("accuracy-history enrichment failed; serving raw")
1240:    return jsonify(entries)
1241:
1242:
1243:@app.route("/api/local-llm-trends")
1244:@require_auth
1245:def api_local_llm_trends():
1246:    """Return local-LLM report trend data for dashboard charts.
1247:
1248:    Query params:
1249:      - limit: number of history points to return (default 90, max 366)
1250:      - ticker: optional ticker filter for Ministral per-ticker series
1251:    """
1252:    limit = _parse_limit_arg("limit", default=90, max_value=366)
1253:    ticker = request.args.get("ticker", "").strip().upper() or None
1254:    latest = _read_json(DATA_DIR / "local_llm_report_latest.json")
1255:    history = _read_jsonl(DATA_DIR / "local_llm_report_history.jsonl", limit=limit)
1256:
1257:    return jsonify({
1258:        "ticker": ticker,
1259:        "latest": latest,
1260:        "series": [
1261:            _build_local_llm_trend_point(entry, ticker=ticker)
1262:            for entry in history
1263:            if isinstance(entry, dict)
1264:        ],
1265:    })
1266:
1267:
1268:@app.route("/api/metals-accuracy")
1269:@require_auth
1270:def api_metals_accuracy():
1271:    """Return metals loop signal accuracy (1h/3h horizons)."""
1272:    data = _read_json(DATA_DIR / "metals_signal_accuracy.json")
1273:    if not data:
1274:        return jsonify({"error": "no data", "stats": {}})
1275:    return jsonify(data)
1276:
1277:
1278:@app.route("/api/trades")
1279:@require_auth
1280:def api_trades():
1281:    """Return combined transactions from both portfolio states for chart annotations."""
1282:    patient = _read_json(DATA_DIR / "portfolio_state.json")
1283:    bold = _read_json(DATA_DIR / "portfolio_state_bold.json")
1284:    trades = []
1285:    if patient and patient.get("transactions"):
1286:        for tx in patient["transactions"]:
1287:            trades.append({
1288:                "ts": tx.get("timestamp", ""),
1289:                "ticker": tx.get("ticker", ""),
1290:                "action": tx.get("action", ""),
1291:                "total_sek": tx.get("total_sek", 0),
1292:                "price_usd": tx.get("price_usd", 0),
1293:                "strategy": "patient",
1294:            })
1295:    if bold and bold.get("transactions"):
1296:        for tx in bold["transactions"]:
1297:            trades.append({
1298:                "ts": tx.get("timestamp", ""),
1299:                "ticker": tx.get("ticker", ""),
1300:                "action": tx.get("action", ""),
1301:                "total_sek": tx.get("total_sek", 0),
1302:                "price_usd": tx.get("price_usd", 0),
1303:                "strategy": "bold",
1304:            })
1305:    trades.sort(key=lambda t: t.get("ts", ""))
1306:    return jsonify(trades)
1307:
1308:
1309:@app.route("/api/decisions")
1310:@require_auth
1311:def api_decisions():
1312:    """Return Layer 2 decision history with optional filtering.
1313:
1314:    Query params:
1315:      - limit: max entries (default 50, max 500)
1316:      - ticker: filter by ticker (e.g., BTC-USD)
1317:      - action: filter by action (BUY, SELL, HOLD)
1318:      - strategy: filter by strategy (patient, bold)
1319:    """
1320:    limit = _parse_limit_arg("limit", default=50, max_value=500)
1321:    ticker_filter = request.args.get("ticker", "").upper()
1322:    action_filter = request.args.get("action", "").upper()
1323:    strategy_filter = request.args.get("strategy", "").lower()
1324:
1325:    results = []
1326:    for entry in _iter_latest_dict_entries(DATA_DIR / "layer2_journal.jsonl", read_limit=1000):
1327:        # Apply action/strategy filters
1328:        if action_filter or strategy_filter:
1329:            decisions = entry.get("decisions", {})
1330:            matched = False
1331:            for strat, dec in decisions.items():
1332:                if strategy_filter and strat != strategy_filter:
1333:                    continue
1334:                if action_filter and dec.get("action", "").upper() != action_filter:
1335:                    continue
1336:                matched = True
1337:            if not matched:
1338:                continue
1339:
1340:        if ticker_filter:
1341:            tickers = entry.get("tickers", {})
1342:            if ticker_filter not in tickers:
1343:                continue
1344:
1345:        results.append(entry)
1346:        if len(results) >= limit:
1347:            break
1348:
1349:    return jsonify(results)
1350:
1351:
1352:@app.route("/api/health")
1353:@require_auth
1354:def api_health():
1355:    """Return system health summary (loop heartbeat, errors, agent silence)."""
1356:    try:
1357:        from portfolio.health import get_health_summary
1358:        return jsonify(get_health_summary())
1359:    except Exception:
1360:        logger.exception("health endpoint error")
1361:        return jsonify({"error": "Internal server error"}), 500
1362:
1363:
1364:# ---------------------------------------------------------------------------
1365:# New: Warrant portfolio
1366:# ---------------------------------------------------------------------------
1367:
1368:@app.route("/api/warrants")
1369:@require_auth
1370:def api_warrants():
1371:    """Return warrant holdings with leverage P&L.
1372:
1373:    Reads data/portfolio_state_warrants.json. Returns empty structure if missing.
1374:    """
1375:    data = _read_json(DATA_DIR / "portfolio_state_warrants.json")
1376:    if not data:
1377:        return jsonify({"holdings": {}, "transactions": []})
1378:    return jsonify(data)
1379:
1380:
1381:# ---------------------------------------------------------------------------
1382:# New: Risk data (Monte Carlo + VaR)
1383:# ---------------------------------------------------------------------------
1384:
1385:@app.route("/api/risk")
1386:@require_auth
1387:def api_risk():
1388:    """Return Monte Carlo price bands and Portfolio VaR from compact summary.
1389:
1390:    Reads monte_carlo and portfolio_var sections from agent_summary_compact.json.
1391:    """
1392:    compact = _read_json(DATA_DIR / "agent_summary_compact.json")
1393:    if not compact:
1394:        return jsonify({"monte_carlo": {}, "portfolio_var": {}})
1395:    return jsonify({
1396:        "monte_carlo": compact.get("monte_carlo", {}),
1397:        "portfolio_var": compact.get("portfolio_var", {}),
1398:    })
1399:
1400:
1401:# ---------------------------------------------------------------------------
1402:# New: Metals monitoring
1403:# ---------------------------------------------------------------------------
1404:
1405:@app.route("/api/metals")
1406:@require_auth
1407:def api_metals():
1408:    """Return combined metals monitoring data.
1409:
1410:    Reads:
1411:      - data/metals_context.json — live positions, P&L, risk, signals, prices
1412:      - data/metals_decisions.jsonl — decision log (newest first, last 50)
1413:      - data/metals_history.json — YTD stats + daily OHLCV
1414:      - data/silver_analysis.json — multi-TF technicals
1415:
1416:    Falls back to the currently-available loop outputs when metals_context.json
1417:    has not been written yet, so the Metals tab still renders partial live data.
1418:    """
1419:    decisions = _normalize_metals_decisions(
1420:        list(_iter_latest_dict_entries(DATA_DIR / "metals_decisions.jsonl", read_limit=50))
1421:    )
1422:    context = _normalize_metals_context(_read_json(DATA_DIR / "metals_context.json"))
1423:    fallback_context = _build_metals_context_fallback(decisions)
1424:    context = _merge_missing_structure(context, fallback_context)
1425:    history = _read_json(DATA_DIR / "metals_history.json")
1426:    technicals = _read_json(DATA_DIR / "silver_analysis.json")
1427:    return jsonify({
1428:        "context": context,
1429:        "decisions": decisions,
1430:        "history": history,
1431:        "technicals": technicals,
1432:    })
1433:
1434:
1435:# ---------------------------------------------------------------------------
1436:# Crypto + MSTR swing-trader endpoints (mirror /api/metals shape)
1437:# ---------------------------------------------------------------------------
1438:
1439:def _crypto_per_instrument(state: dict, ticker: str) -> dict:
1440:    """Slice the unified crypto_swing_state.json by ticker."""
1441:    positions = state.get("positions", {}) if state else {}
1442:    matches = {pid: p for pid, p in positions.items() if p.get("ticker") == ticker}
1443:    return {
1444:        "n_positions": len(matches),
1445:        "positions": matches,
1446:        "last_buy_ts": (state.get("last_buy_ts", {}) or {}).get(ticker)
1447:                       if state else None,
1448:    }
1449:
1450:
1451:def _crypto_decisions_for(decisions: list, ticker: str) -> list:
1452:    out = []
1453:    for d in decisions or []:
1454:        pos = d.get("pos") or {}
1455:        if pos.get("ticker") == ticker:
1456:            out.append(d)
1457:        elif d.get("ticker") == ticker:
1458:            out.append(d)
1459:    return out
1460:
1461:
1462:@app.route("/api/crypto")
1463:@require_auth
1464:def api_crypto():
1465:    """Combined BTC + ETH swing-trader state (mirror of /api/metals).
1466:
1467:    Reads:
1468:      - data/crypto_swing_state.json (positions, cash, cycle counter)
1469:      - data/crypto_deep_context.json (Fear & Greed, funding, on-chain)
1470:      - data/crypto_swing_decisions.jsonl (last 50)
1471:      - data/crypto_swing_trades.jsonl (last 50)
1472:      - data/crypto_warrant_catalog.json (live warrant universe)
1473:      - data/crypto_risk.json (per-position barrier checks, drawdown)
1474:    """
1475:    state = _read_json(DATA_DIR / "crypto_swing_state.json") or {}
1476:    context = _read_json(DATA_DIR / "crypto_deep_context.json") or {}
1477:    catalog = _read_json(DATA_DIR / "crypto_warrant_catalog.json") or {}
1478:    risk = _read_json(DATA_DIR / "crypto_risk.json") or {}
1479:    decisions = list(_iter_latest_dict_entries(
1480:        DATA_DIR / "crypto_swing_decisions.jsonl", read_limit=50))
1481:    trades = list(_iter_latest_dict_entries(
1482:        DATA_DIR / "crypto_swing_trades.jsonl", read_limit=50))
1483:    return jsonify({
1484:        "state": state,
1485:        "context": context,
1486:        "warrant_catalog": catalog,
1487:        "risk": risk,
1488:        "decisions": decisions,
1489:        "trades": trades,
1490:    })
1491:
1492:
1493:@app.route("/api/btc")
1494:@require_auth
1495:def api_btc():
1496:    """BTC-specific slice of the crypto swing-trader state."""
1497:    state = _read_json(DATA_DIR / "crypto_swing_state.json") or {}
1498:    context = _read_json(DATA_DIR / "crypto_deep_context.json") or {}
1499:    decisions = list(_iter_latest_dict_entries(
1500:        DATA_DIR / "crypto_swing_decisions.jsonl", read_limit=50))
1501:    trades = list(_iter_latest_dict_entries(
1502:        DATA_DIR / "crypto_swing_trades.jsonl", read_limit=50))
1503:    return jsonify({
1504:        "ticker": "BTC-USD",
1505:        "instrument": _crypto_per_instrument(state, "BTC-USD"),
1506:        "deep_context": (context or {}).get("btc"),
1507:        "shared_context": (context or {}).get("shared"),
1508:        "decisions": _crypto_decisions_for(decisions, "BTC-USD"),
1509:        "trades": _crypto_decisions_for(trades, "BTC-USD"),
1510:    })
1511:
1512:
1513:@app.route("/api/eth")
1514:@require_auth
1515:def api_eth():
1516:    """ETH-specific slice of the crypto swing-trader state."""
1517:    state = _read_json(DATA_DIR / "crypto_swing_state.json") or {}
1518:    context = _read_json(DATA_DIR / "crypto_deep_context.json") or {}
1519:    decisions = list(_iter_latest_dict_entries(
1520:        DATA_DIR / "crypto_swing_decisions.jsonl", read_limit=50))
1521:    trades = list(_iter_latest_dict_entries(
1522:        DATA_DIR / "crypto_swing_trades.jsonl", read_limit=50))
1523:    return jsonify({
1524:        "ticker": "ETH-USD",
1525:        "instrument": _crypto_per_instrument(state, "ETH-USD"),
1526:        "deep_context": (context or {}).get("eth"),
1527:        "shared_context": (context or {}).get("shared"),
1528:        "decisions": _crypto_decisions_for(decisions, "ETH-USD"),
1529:        "trades": _crypto_decisions_for(trades, "ETH-USD"),
1530:    })
1531:
1532:
1533:@app.route("/api/loop_health")
1534:@require_auth
1535:def api_loop_health():
1536:    """Cross-loop heartbeat rollup.
1537:
1538:    Reads data/{name}_loop.heartbeat for each registered loop (currently
1539:    crypto + oil; metals/main can be added when they grow heartbeats).
1540:    Returns per-loop {state, age_seconds, payload, error}, plus a
1541:    rollup any_unhealthy flag and an unhealthy[] list.
1542:
1543:    Same data the loop-health watchdog uses for telegram alerts. Use
1544:    this endpoint for live dashboard monitoring without waiting for the
1545:    next watchdog tick.
1546:    """
1547:    from portfolio.loop_health import read_loop_health
1548:    return jsonify(read_loop_health())
1549:
1550:
1551:@app.route("/api/oil")
1552:@require_auth
1553:def api_oil():
1554:    """Oil swing-trader state (mirror of /api/crypto and /api/metals).
1555:
1556:    Reads:
1557:      - data/oil_swing_state.json (positions, cash, cycle counter)
1558:      - data/oil_deep_context.json (WTI/Brent/COT/OVX/crack-spread context
1559:        from portfolio/oil_precompute.py)
1560:      - data/oil_swing_decisions.jsonl (last 50)
1561:      - data/oil_swing_trades.jsonl (last 50)
1562:      - data/oil_warrant_catalog.json (live OLJA warrant universe)
1563:      - data/oil_risk.json (per-position barrier checks, drawdown)
1564:
1565:    Ships in DRY_RUN=True; the trades log will be empty until the loop
1566:    is wired live via data/oil_swing_config.py.
1567:    """
1568:    state = _read_json(DATA_DIR / "oil_swing_state.json") or {}
1569:    context = _read_json(DATA_DIR / "oil_deep_context.json") or {}
1570:    catalog = _read_json(DATA_DIR / "oil_warrant_catalog.json") or {}
1571:    risk = _read_json(DATA_DIR / "oil_risk.json") or {}
1572:    decisions = list(_iter_latest_dict_entries(
1573:        DATA_DIR / "oil_swing_decisions.jsonl", read_limit=50))
1574:    trades = list(_iter_latest_dict_entries(
1575:        DATA_DIR / "oil_swing_trades.jsonl", read_limit=50))
1576:    # Heartbeat reflects liveness even when no trades have fired
1577:    heartbeat = _read_json(DATA_DIR / "oil_loop.heartbeat") or {}
1578:    return jsonify({
1579:        "state": state,
1580:        "context": context,
1581:        "warrant_catalog": catalog,
1582:        "risk": risk,
1583:        "decisions": decisions,
1584:        "trades": trades,
1585:        "heartbeat": heartbeat,
1586:    })
1587:
1588:
1589:@app.route("/api/mstr")
1590:@require_auth
1591:def api_mstr():
1592:    """MSTR deep-context endpoint.
1593:
1594:    The pre-existing `/api/mstr_loop` returns the strategy-loop state
1595:    (positions, scorecard, last poll). This new endpoint returns the deep
1596:    context (NAV premium, BTC correlation, options skew, analyst consensus)
1597:    written by `portfolio/mstr_precompute.py`. Together they parallel
1598:    `/api/metals` (decisions+context) for the metals subsystem.
1599:    """
1600:    deep = _read_json(DATA_DIR / "mstr_deep_context.json") or {}
1601:    loop_state = _read_json(DATA_DIR / "mstr_loop_state.json") or {}
1602:    scorecard = _read_json(DATA_DIR / "mstr_loop_scorecard.json") or {}
1603:    return jsonify({
1604:        "ticker": "MSTR",
1605:        "deep_context": deep,
1606:        "loop_state": loop_state,
1607:        "scorecard": scorecard,
1608:    })
1609:
1610:
1611:# ---------------------------------------------------------------------------
1612:# New: GoldDigger monitoring
1613:# ---------------------------------------------------------------------------
1614:
1615:@app.route("/api/golddigger")
1616:@require_auth
1617:def api_golddigger():
1618:    """Return GoldDigger signal data normalized for the dashboard.
1619:
1620:    The bot persists a lean state snapshot plus compact JSONL logs. This route
1621:    reshapes those records into the richer schema expected by the dashboard UI.
1622:    """
1623:    raw_log = list(_iter_latest_dict_entries(DATA_DIR / "golddigger_log.jsonl", read_limit=100))
1624:    raw_trades = list(_iter_latest_dict_entries(DATA_DIR / "golddigger_trades.jsonl", read_limit=50))
1625:    state = _normalize_golddigger_state(_read_json(DATA_DIR / "golddigger_state.json"), raw_log)
1626:    log = [entry for entry in (_normalize_golddigger_log_entry(item) for item in raw_log) if entry]
1627:    trades = [entry for entry in (_normalize_golddigger_trade_entry(item) for item in raw_trades) if entry]
1628:    return jsonify({
1629:        "state": state if state or log or trades else None,
1630:        "log": log,
1631:        "trades": trades,
1632:    })
1633:
1634:
1635:# ---------------------------------------------------------------------------
1636:# Market health
1637:# ---------------------------------------------------------------------------
1638:
1639:@app.route("/api/market-health")
1640:@require_auth
1641:def api_market_health():
1642:    """Return market health snapshot (distribution days, FTD, breadth score).
1643:
1644:    Also includes exposure recommendation and earnings proximity data.
1645:    """
1646:    try:
1647:        result = {}
1648:        # Market health from agent_summary (pre-computed) or live
1649:        summary = _read_json(DATA_DIR / "agent_summary.json")
1650:        if summary and "market_health" in summary:
1651:            result["market_health"] = summary["market_health"]
1652:        else:
1653:            try:
1654:                from portfolio.market_health import get_market_health
1655:                mh = get_market_health()
1656:                if mh:
1657:                    result["market_health"] = mh
1658:            except Exception:
1659:                # BUG-205: log at debug so a broken market_health source is
1660:                # diagnosable instead of silently omitting the field.
1661:                logger.debug("market_health enrichment failed", exc_info=True)
1662:
1663:        if summary:
1664:            if "exposure_recommendation" in summary:
1665:                result["exposure_recommendation"] = summary["exposure_recommendation"]
1666:            if "earnings_proximity" in summary:
1667:                result["earnings_proximity"] = summary["earnings_proximity"]
1668:
1669:        return jsonify(result)
1670:    except Exception:
1671:        logger.exception("mstr endpoint error")
1672:        return jsonify({"error": "Internal server error"}), 500
1673:
1674:
1675:# ---------------------------------------------------------------------------
1676:# Avanza account snapshot — live cash + positions + open orders + stop-losses.
1677:# Lets the user verify the local view is in sync with the actual broker
1678:# state. Each subsection is independently fetched so a single API hiccup
1679:# (e.g. flaky stop-loss endpoint) doesn't blank the whole view.
1680:# ---------------------------------------------------------------------------
1681:
1682:_AVANZA_CACHE_LOCK = threading.Lock()
1683:_AVANZA_CACHE: dict[str, Any] = {"at": 0.0, "value": None}
1684:_AVANZA_TTL_SECONDS = 30.0
1685:
1686:# Same TTL pattern for the system-health rollup endpoints. Both caches
1687:# are independent so trading_status can refresh on its own cadence
1688:# while system_status keeps serving cached, and vice versa.
1689:_SYSTEM_STATUS_LOCK = threading.Lock()
1690:_SYSTEM_STATUS_CACHE: dict[str, Any] = {"at": 0.0, "value": None}
1691:_SYSTEM_STATUS_TTL_SECONDS = 30.0
1692:
1693:_TRADING_STATUS_LOCK = threading.Lock()
1694:_TRADING_STATUS_CACHE: dict[str, Any] = {"at": 0.0, "value": None}
1695:_TRADING_STATUS_TTL_SECONDS = 30.0
1696:
1697:# ---------------------------------------------------------------------------
1698:# Avanza worker thread — Playwright's sync API is bound to its creator
1699:# thread, but Flask's ThreadedWSGIServer spawns a fresh worker per request.
1700:# A request that lands on a different thread than the one which initialised
1701:# Playwright fails with "cannot switch to a different thread (which happens
1702:# to have exited)".
1703:#
1704:# Solution: a single dedicated worker thread owns the Playwright session
1705:# for the dashboard process. HTTP handlers enqueue snapshot requests via
1706:# `_avanza_request_q`, the worker processes them in order, and replies via
1707:# a per-request Event. This is the same pattern the metals_loop dodges by
1708:# being single-threaded; Flask can't afford that, so we serialise here.
1709:# ---------------------------------------------------------------------------
1710:
1711:import queue  # noqa: E402  (kept near the worker for grouping)
1712:
1713:_AVANZA_REQ_Q: "queue.Queue[dict]" = queue.Queue()
1714:_AVANZA_WORKER_LOCK = threading.Lock()
1715:_AVANZA_WORKER_STARTED = False
1716:_AVANZA_REQ_TIMEOUT_SECONDS = 25.0  # snapshot upper bound
1717:
1718:
1719:def _avanza_worker_loop() -> None:
1720:    """Single-thread worker that owns Playwright. Blocks on the request
1721:    queue and serves snapshot requests sequentially."""
1722:    while True:
1723:        future = _AVANZA_REQ_Q.get()
1724:        try:
1725:            future["result"] = _avanza_snapshot_impl()
1726:        except Exception as e:
1727:            logger.exception("avanza-worker: snapshot failed")
1728:            future["result"] = {
1729:                "ts": datetime.now(UTC).isoformat(),
1730:                "account_id": None,
1731:                "cash": None,
1732:                "positions": [],
1733:                "orders": [],
1734:                "stop_losses": [],
1735:                "errors": [f"worker: {type(e).__name__}: {e}"],
1736:            }
1737:        finally:
1738:            future["done"].set()
1739:
1740:
1741:def _ensure_avanza_worker() -> None:
1742:    global _AVANZA_WORKER_STARTED
1743:    if _AVANZA_WORKER_STARTED:
1744:        return
1745:    with _AVANZA_WORKER_LOCK:
1746:        if _AVANZA_WORKER_STARTED:
1747:            return
1748:        t = threading.Thread(
1749:            target=_avanza_worker_loop, daemon=True, name="avanza-worker",
1750:        )
1751:        t.start()
1752:        _AVANZA_WORKER_STARTED = True
1753:
1754:
1755:def _avanza_account_snapshot() -> dict:
1756:    """Public entry. Marshals snapshot building onto the worker thread so
1757:    Playwright's thread affinity is honoured."""
1758:    _ensure_avanza_worker()
1759:    future: dict[str, Any] = {"result": None, "done": threading.Event()}
1760:    _AVANZA_REQ_Q.put(future)
1761:    if not future["done"].wait(timeout=_AVANZA_REQ_TIMEOUT_SECONDS):
1762:        return {
1763:            "ts": datetime.now(UTC).isoformat(),
1764:            "account_id": None,
1765:            "cash": None,
1766:            "positions": [],
1767:            "orders": [],
1768:            "stop_losses": [],
1769:            "errors": [
1770:                f"avanza-worker: timed out after {_AVANZA_REQ_TIMEOUT_SECONDS}s"
1771:            ],
1772:        }
1773:    return future["result"] or {
1774:        "ts": datetime.now(UTC).isoformat(),
1775:        "account_id": None, "cash": None, "positions": [],
1776:        "orders": [], "stop_losses": [],
1777:        "errors": ["avanza-worker: empty result"],
1778:    }
1779:
1780:
1781:def _avanza_snapshot_impl() -> dict:
1782:    """Build a fresh Avanza account snapshot. Uncached.
1783:
1784:    Uses `portfolio.avanza_session` (Playwright BankID auth at
1785:    `data/avanza_session.json`) — the same path the live metals_loop and
1786:    golddigger use. The newer `portfolio.avanza` TOTP package is *not*
1787:    used here because TOTP credentials aren't populated in the live
1788:    config; switching needs setup work outside this PR. Codex P1 fix
1789:    2026-05-04 originally seeded the TOTP singleton, but the empty
1790:    credentials made every call still fail — the live-system path is
1791:    the right answer.
1792:
1793:    Each subcall is independently try/except'd so a partial Avanza
1794:    outage degrades section-by-section. Sections are filtered to the
1795:    configured account_id (codex P2 finding 2026-05-04).
1796:    """
1797:    out: dict[str, Any] = {
1798:        "ts": datetime.now(UTC).isoformat(),
1799:        "account_id": None,
1800:        "cash": None,
1801:        "positions": [],
1802:        "orders": [],
1803:        "stop_losses": [],
1804:        "errors": [],
1805:    }
1806:    try:
1807:        from portfolio.avanza_session import DEFAULT_ACCOUNT_ID
1808:        account_id = str(DEFAULT_ACCOUNT_ID)
1809:    except Exception:
1810:        account_id = None
1811:    out["account_id"] = account_id
1812:
1813:    try:
1814:        from portfolio.avanza_session import get_buying_power
1815:        cash = get_buying_power(account_id=account_id)
1816:        if cash is None:
1817:            out["errors"].append(
1818:                "cash: get_buying_power returned None "
1819:                "(Avanza session likely expired — re-auth via BankID)"
1820:            )
1821:        else:
1822:            out["cash"] = cash
1823:    except Exception as e:
1824:        out["errors"].append(f"cash: {type(e).__name__}: {e}")
1825:
1826:    try:
1827:        from portfolio.avanza_session import get_positions
1828:        all_positions = get_positions()
1829:        out["positions"] = [
1830:            p for p in all_positions
1831:            if account_id is None or str(p.get("account_id", "")) == account_id
1832:        ]
1833:    except Exception as e:
1834:        out["errors"].append(f"positions: {type(e).__name__}: {e}")
1835:
1836:    try:
1837:        from portfolio.avanza_session import get_open_orders
1838:        out["orders"] = [_norm_order(o) for o in get_open_orders(account_id=account_id)]
1839:    except Exception as e:
1840:        out["errors"].append(f"orders: {type(e).__name__}: {e}")
1841:
1842:    try:
1843:        from portfolio.avanza_session import get_stop_losses
1844:        stops = get_stop_losses()
1845:        out["stop_losses"] = [
1846:            _norm_stop(s) for s in stops
1847:            if account_id is None or str(_stop_account(s)) == account_id
1848:        ]
1849:    except Exception as e:
1850:        out["errors"].append(f"stop_losses: {type(e).__name__}: {e}")
1851:    return out
1852:
1853:
1854:def _norm_order(raw: dict) -> dict:
1855:    """Normalize an Avanza orders-API dict to the snake_case shape the
1856:    dashboard view binds against."""
1857:    return {
1858:        "order_id":     str(raw.get("orderId", raw.get("id", ""))),
1859:        "orderbook_id": str(raw.get("orderBookId", raw.get("orderbookId", ""))),
1860:        "side":         str(raw.get("orderType", raw.get("side", ""))),
1861:        "price":        float(raw.get("price") or 0.0),
1862:        "volume":       int(raw.get("volume") or 0),
1863:        "status":       str(raw.get("status", raw.get("statusDescription", ""))),
1864:        "account_id":   str(raw.get("accountId", raw.get("account_id", ""))),
1865:    }
1866:
1867:
1868:def _stop_account(raw: dict) -> str:
1869:    return str(
1870:        raw.get("accountId") or raw.get("account_id") or
1871:        (raw.get("account") or {}).get("id", "")
1872:    )
1873:
1874:
1875:def _norm_stop(raw: dict) -> dict:
1876:    """Normalize an Avanza stop-loss dict (matches Order.from_api shape)."""
1877:    trigger = raw.get("trigger") or {}
1878:    order_event = raw.get("orderEvent") or raw.get("order") or {}
1879:    return {
1880:        "stop_id":       str(raw.get("id", raw.get("stopLossId", ""))),
1881:        "orderbook_id":  str((raw.get("orderbook") or {}).get("id",
1882:                              raw.get("orderBookId", raw.get("orderbookId", "")))),
1883:        "trigger_price": float(trigger.get("value") or raw.get("triggerPrice") or 0.0),
1884:        "trigger_type":  str(trigger.get("type") or raw.get("triggerType") or "LAST_PRICE"),
1885:        "sell_price":    float(order_event.get("price") or raw.get("sellPrice") or 0.0),
1886:        "volume":        int(order_event.get("volume") or raw.get("volume") or 0),
1887:        "status":        str(raw.get("status", "")),
1888:        "account_id":    _stop_account(raw),
1889:    }
1890:
1891:
1892:@app.route("/api/avanza_account")
1893:@require_auth
1894:def api_avanza_account():
1895:    """Live snapshot of the Avanza brokerage account.
1896:
1897:    Cash + positions + open orders + active stop-losses, filtered to the
1898:    configured account_id. 30-second TTL cache because the underlying
1899:    calls hit the network. Each subsection has its own try/except so a
1900:    partial upstream outage degrades to "this section unavailable"
1901:    instead of a full 500.
1902:
1903:    `?force=1` bypasses the TTL cache so the user's manual Refresh
1904:    button can verify a just-placed or cancelled order without waiting
1905:    out the polling cadence (Codex P2 finding 2026-05-04).
1906:    """
1907:    force = request.args.get("force", "").strip() in {"1", "true", "yes"}
1908:    now = time.monotonic()
1909:    if not force:
1910:        with _AVANZA_CACHE_LOCK:
1911:            cached = _AVANZA_CACHE.get("value")
1912:            if cached and (now - _AVANZA_CACHE["at"]) < _AVANZA_TTL_SECONDS:
1913:                return jsonify(cached)
1914:    snapshot = _avanza_account_snapshot()
1915:    with _AVANZA_CACHE_LOCK:
1916:        _AVANZA_CACHE["value"] = snapshot
1917:        _AVANZA_CACHE["at"] = now
1918:    return jsonify(snapshot)
1919:
1920:
1921:# ---------------------------------------------------------------------------
1922:# Tradeable assets — what the loops will buy/sell. Aggregates the metals
1923:# warrant catalog (fin_fish), crypto + oil JSON catalogs, plus the small
1924:# equity universe in avanza_tracker. Lets the user verify the system
1925:# knows about each instrument, including its orderbook_id, leverage, and
1926:# direction. Read-only.
1927:# ---------------------------------------------------------------------------
1928:
1929:@app.route("/api/tradeable_assets")
1930:@require_auth
1931:def api_tradeable_assets():
1932:    """Return everything the system might trade on Avanza.
1933:
1934:    Aggregates:
1935:      - Metals warrants (`portfolio.fin_fish.WARRANT_CATALOG`)
1936:      - Crypto warrants (`data/crypto_warrant_catalog.json`)
1937:      - Oil warrants (`data/oil_warrant_catalog.json`)
1938:
1939:    Each category is independently try/except'd so a missing import or
1940:    bad JSON file doesn't blank the whole view.
1941:    """
1942:    out: dict[str, Any] = {
1943:        "ts": datetime.now(UTC).isoformat(),
1944:        "metals_warrants": {},
1945:        "crypto_warrants": {},
1946:        "oil_warrants": {},
1947:        "errors": [],
1948:    }
1949:    try:
1950:        from portfolio.fin_fish import WARRANT_CATALOG as METALS_CATALOG
1951:        out["metals_warrants"] = dict(METALS_CATALOG)
1952:    except Exception as e:
1953:        out["errors"].append(f"metals: {type(e).__name__}: {e}")
1954:    try:
1955:        crypto = _read_json(DATA_DIR / "crypto_warrant_catalog.json") or {}
1956:        out["crypto_warrants"] = crypto.get("warrants", crypto) if isinstance(crypto, dict) else {}
1957:    except Exception as e:
1958:        out["errors"].append(f"crypto: {type(e).__name__}: {e}")
1959:    try:
1960:        oil = _read_json(DATA_DIR / "oil_warrant_catalog.json") or {}
1961:        out["oil_warrants"] = oil.get("warrants", oil) if isinstance(oil, dict) else {}
1962:    except Exception as e:
1963:        out["errors"].append(f"oil: {type(e).__name__}: {e}")
1964:    return jsonify(out)
1965:
1966:
1967:# ---------------------------------------------------------------------------
1968:# System-health home rollup endpoints.
1969:#
1970:# /api/system_status   - overall GREEN/YELLOW/RED, heartbeat, errors,
1971:#                        contract violations, LLM inference success,
1972:#                        Layer 2 24h activity, signal aggregate.
1973:# /api/trading_status  - per-bot Avanza state with reason
1974:#                        (golddigger, elongir, metals, fishing).
1975:#
1976:# Both are pure aggregations over data/*.json[l]. No network. 30s TTL
1977:# cache mirrors the _AVANZA_CACHE pattern; ?force=1 bypasses for the
1978:# manual Refresh button.
1979:# ---------------------------------------------------------------------------
1980:
1981:@app.route("/api/system_status")
1982:@require_auth
1983:def api_system_status():
1984:    """System-health rollup for the home view's GREEN/YELLOW/RED hero.
1985:
1986:    See dashboard/system_status.py for the full payload shape and
1987:    severity thresholds. Per-section errors[] envelope so a corrupt
1988:    jsonl line never blanks the hero.
1989:
1990:    Cache discipline (codex P2 finding 2026-05-04): the lock covers
1991:    both the read and the write so concurrent misses serialize. A
1992:    request that started after the most recent fill won't overwrite a
1993:    fresher payload, and ``?force=1`` won't lose its refresh behind
1994:    another in-flight fill.
1995:    """
1996:    from dashboard import system_status as _sys_status
1997:
1998:    force = request.args.get("force", "").strip() in {"1", "true", "yes"}
1999:    if not force:
2000:        with _SYSTEM_STATUS_LOCK:
2001:            cached = _SYSTEM_STATUS_CACHE.get("value")
2002:            if cached and (time.monotonic() - _SYSTEM_STATUS_CACHE["at"]) < _SYSTEM_STATUS_TTL_SECONDS:
2003:                return jsonify(cached)
2004:    with _SYSTEM_STATUS_LOCK:
2005:        # Re-check inside the lock — a concurrent miss may have filled it.
2006:        cached = _SYSTEM_STATUS_CACHE.get("value")
2007:        if not force and cached and (time.monotonic() - _SYSTEM_STATUS_CACHE["at"]) < _SYSTEM_STATUS_TTL_SECONDS:
2008:            return jsonify(cached)
2009:        payload = _sys_status.compute()
2010:        _SYSTEM_STATUS_CACHE["value"] = payload
2011:        _SYSTEM_STATUS_CACHE["at"] = time.monotonic()
2012:        return jsonify(payload)
2013:
2014:
2015:@app.route("/api/trading_status")
2016:@require_auth
2017:def api_trading_status():
2018:    """Per-bot Avanza trading state with reason.
2019:
2020:    See dashboard/trading_status.py. Each bot resolves to one of
2021:    SCANNING / TRADING / HALTED / COOLDOWN / OUTSIDE_HOURS / UNKNOWN.
2022:    Same lock discipline as ``/api/system_status``.
2023:    """
2024:    from dashboard import trading_status as _trading_status
2025:
2026:    force = request.args.get("force", "").strip() in {"1", "true", "yes"}
2027:    if not force:
2028:        with _TRADING_STATUS_LOCK:
2029:            cached = _TRADING_STATUS_CACHE.get("value")
2030:            if cached and (time.monotonic() - _TRADING_STATUS_CACHE["at"]) < _TRADING_STATUS_TTL_SECONDS:
2031:                return jsonify(cached)
2032:    with _TRADING_STATUS_LOCK:
2033:        cached = _TRADING_STATUS_CACHE.get("value")
2034:        if not force and cached and (time.monotonic() - _TRADING_STATUS_CACHE["at"]) < _TRADING_STATUS_TTL_SECONDS:
2035:            return jsonify(cached)
2036:        payload = _trading_status.compute()
2037:        _TRADING_STATUS_CACHE["value"] = payload
2038:        _TRADING_STATUS_CACHE["at"] = time.monotonic()
2039:        return jsonify(payload)
2040:
2041:
2042:# ---------------------------------------------------------------------------
2043:# Blueprint: /house — read-only viewer over the househunting project
2044:# (data/findapartments runs + innerstad heatmap). Reuses pf_dashboard_token
2045:# auth via dashboard.auth.require_auth. Path roots come from
2046:# config.json[house_root]. See dashboard/house_blueprint.py for routes.
2047:#
2048:# House_blueprint imports `_get_config` and `require_auth` from
2049:# dashboard.auth (NOT dashboard.app), so importing it here at module-init
2050:# time no longer causes a circular import — auth.py has no back-reference
2051:# to app.py. The sys.modules alias hack added 2026-05-02 has been removed.
2052:# ---------------------------------------------------------------------------
2053:from dashboard.house_blueprint import bp as _house_bp  # noqa: E402
2054:
2055:app.register_blueprint(_house_bp)
2056:
2057:
2058:def _serve_dual_stack(port: int = 5055) -> None:
2059:    """Run the Flask app on a dual-stack IPv4+IPv6 socket.
2060:
2061:    2026-05-04: previously used `app.run(host="0.0.0.0", ...)` which is
2062:    IPv4-only. Local Python tooling (urllib, requests) on Windows that
2063:    resolves "localhost" to ::1 first then waits ~2s for the IPv6
2064:    connection to fail before falling back to IPv4 — perceived as a
2065:    universal "2s auth floor" but actually a client-side Happy Eyeballs
2066:    timeout. Real users (Cloudflare tunnel, LAN browsers) never see it.
2067:
2068:    Switching to `host="::"` would fix localhost on Linux but on
2069:    Windows the default `IPV6_V6ONLY=True` socket option means IPv4
2070:    clients can no longer connect. So we bind manually with
2071:    `IPV6_V6ONLY=0`, which works on every modern Windows / Linux /
2072:    macOS host.
2073:    """
2074:    import socket
2075:    from werkzeug.serving import ThreadedWSGIServer
2076:
2077:    # Build the dual-stack listening socket explicitly. IPV6_V6ONLY=0
2078:    # enables IPv4 mapping (::ffff:127.0.0.1 etc.), so a single AF_INET6
2079:    # socket accepts both IPv4 and IPv6 clients.
2080:    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
2081:    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
2082:    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
2083:    sock.bind(("::", port))
2084:    sock.listen(128)
2085:
2086:    # ThreadedWSGIServer accepts `fd=` so it skips its own bind/listen
2087:    # and reuses our pre-configured socket. ThreadingMixIn handles
2088:    # concurrent requests just like Werkzeug's default app.run().
2089:    server = ThreadedWSGIServer("::", port, app, fd=sock.fileno())
2090:    print(f"Dashboard listening on dual-stack [::]:{port} (IPv4 + IPv6)",
2091:          flush=True)
2092:    server.serve_forever()
2093:
2094:
2095:if __name__ == "__main__":
2096:    _serve_dual_stack(port=5055)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 1114ms:
1:"""System-health rollup for the dashboard home page.
2:
3:Pure aggregator. Reads the same jsonl/json files the rest of the system
4:already writes — adds no new instrumentation. Returns a single payload
5:shape that the home view consumes via ``/api/system_status``.
6:
7:Why this exists: the previous home page led with simulated-portfolio
8:P&L, which the user explicitly deprioritised in favour of "is the
9:system actually working" indicators. See
10:``docs/PLAN.md`` 2026-05-04 entry and the plan file
11:``/root/.claude/plans/merry-tinkering-cake.md`` for the full design.
12:
13:Side-effect-free: never opens a network socket, never invokes Avanza,
14:never writes a file. Safe to call from a 30s polling loop.
15:"""
16:
17:from __future__ import annotations
18:
19:from datetime import datetime, timedelta, timezone
20:from pathlib import Path
21:from typing import Any
22:
23:import re
24:
25:from portfolio.file_utils import load_json, load_jsonl, load_jsonl_tail
26:from portfolio.loop_contract import violation_identity_payload
27:
28:# Mirror loop_contract._ESCALATED_PREFIX_RE so the dashboard strips the
29:# tracker's "ESCALATED (Nx consecutive): " prefix before computing the
30:# identity payload — without this, a tracker-promoted CV row hashes
31:# differently from its critical_errors counterpart and the cross-stream
32:# resolution check fails to match. (Claude review of a85a646f, P1-2.)
33:_ESCALATED_PREFIX_RE = re.compile(r"^ESCALATED \(\d+x consecutive\): ")
34:
35:# Repo data dir. Resolved relative to this file so the module works in
36:# both the main checkout and a worktree without further config.
37:DATA_DIR = Path(__file__).resolve().parent.parent / "data"
38:
39:# Severity thresholds — tune freely, no schema migration needed.
40:HEARTBEAT_GREEN_S = 120
41:HEARTBEAT_YELLOW_S = 600
42:ERRORS_YELLOW_MAX = 3
43:LLM_GREEN_PCT = 95.0
44:LLM_YELLOW_PCT = 80.0
45:LAYER2_GREEN_PCT = 85.0
46:LAYER2_YELLOW_PCT = 60.0
47:LAYER2_MIN_TRIGGERS_FOR_GATE = 3
48:
49:UTC = timezone.utc
50:
51:
52:# ---------------------------------------------------------------------------
53:# Public entry
54:# ---------------------------------------------------------------------------
55:
56:
57:def compute(data_dir: Path | None = None) -> dict[str, Any]:
58:    """Build the full system_status payload.
59:
60:    Each section catches its own exceptions and surfaces an ``error``
61:    string in-band so a single bad jsonl line doesn't blank the hero.
62:    Mirrors the per-section error envelope used by ``/api/avanza_account``.
63:    """
64:    dd = Path(data_dir) if data_dir else DATA_DIR
65:    out: dict[str, Any] = {
66:        "ts": datetime.now(UTC).isoformat(),
67:        "heartbeat": _heartbeat(dd),
68:        "errors": _errors_unresolved(dd),
69:        "contract_violations": _violations_recent(dd),
70:        "llm_inference": _llm_inference(dd),
71:        "layer2": _layer2_24h(dd),
72:        "signal_aggregate": _signal_aggregate(dd),
73:        "pnl_footer": _pnl_footer(dd),
74:    }
75:    overall, reasons = _color(out)
76:    out["overall"] = overall
77:    out["reasons"] = reasons
78:    return out
79:
80:
81:# ---------------------------------------------------------------------------
82:# Sections
83:# ---------------------------------------------------------------------------
84:
85:
86:def _heartbeat(dd: Path) -> dict[str, Any]:
87:    """Codex P2 follow-up: outer try/except so any parse / I/O failure
88:    surfaces in-band rather than 500'ing the whole endpoint."""
89:    try:
90:        health = load_json(dd / "health_state.json", default={}) or {}
91:        if not isinstance(health, dict):
92:            return _hb_default(error="health_state.json is not a JSON object")
93:        last = health.get("last_heartbeat")
94:        age_s: float | None = None
95:        err: str | None = None
96:        if last:
97:            try:
98:                age_s = (datetime.now(UTC) - _parse_ts(str(last))).total_seconds()
99:            except Exception as e:
100:                err = f"heartbeat parse: {type(e).__name__}: {e}"
101:        out: dict[str, Any] = {
102:            "age_seconds": age_s,
103:            "last_ts": last,
104:            "cycle_count": health.get("cycle_count", 0),
105:            "error_count": health.get("error_count", 0),
106:        }
107:        if err:
108:            out["error"] = err
109:        return out
110:    except Exception as e:
111:        return _hb_default(error=f"heartbeat: {type(e).__name__}: {e}")
112:
113:
114:def _hb_default(error: str | None = None) -> dict[str, Any]:
115:    out: dict[str, Any] = {
116:        "age_seconds": None,
117:        "last_ts": None,
118:        "cycle_count": 0,
119:        "error_count": 0,
120:    }
121:    if error:
122:        out["error"] = error
123:    return out
124:
125:
126:def _errors_unresolved(dd: Path) -> dict[str, Any]:
127:    """Walk critical_errors.jsonl. An entry with category="resolution"
128:    and ``resolves_ts`` pointing at an earlier entry resolves it.
129:
130:    Codex P1 finding 2026-05-04: this MUST scan the whole file, not a
131:    fixed tail. If 500 newer info/resolution rows came after older
132:    unresolved criticals, the older ones disappeared from the count
133:    and the home page silently flipped to GREEN. critical_errors.jsonl
134:    is small (~120 KB at the time of writing); we accept the full scan
135:    behind the 30s TTL cache.
136:    """
137:    try:
138:        entries = load_jsonl(dd / "critical_errors.jsonl")
139:    except Exception as e:
140:        return {"unresolved": 0, "recent": [], "error": f"errors load: {type(e).__name__}: {e}"}
141:    try:
142:        resolved_ts: set[str] = set()
143:        by_ts: dict[str, dict] = {}
144:        for entry in entries:
145:            if not isinstance(entry, dict):
146:                continue
147:            ts = entry.get("ts")
148:            if entry.get("category") == "resolution" and entry.get("resolves_ts"):
149:                resolved_ts.add(str(entry["resolves_ts"]))
150:                continue
151:            # Skip non-critical levels (resolution rows arrive as "info").
152:            if entry.get("level") and entry.get("level") != "critical":
153:                continue
154:            if ts:
155:                by_ts[ts] = entry
156:        unresolved = [e for ts, e in by_ts.items() if ts not in resolved_ts]
157:        unresolved.sort(key=lambda x: x.get("ts", ""), reverse=True)
158:        recent = [
159:            {
160:                "ts": e.get("ts"),
161:                "category": e.get("category"),
162:                "caller": e.get("caller"),
163:                "message": (e.get("message") or "")[:200],
164:            }
165:            for e in unresolved[:5]
166:        ]
167:        return {"unresolved": len(unresolved), "recent": recent}
168:    except Exception as e:
169:        return {"unresolved": 0, "recent": [], "error": f"errors aggregate: {type(e).__name__}: {e}"}
170:
171:
172:def _violations_recent(dd: Path) -> dict[str, Any]:
173:    """Last-24h CRITICAL contract violations, filtered to *unresolved* rows.
174:
175:    Uses adaptive tail growth so we don't undercount on high-volume
176:    days. ``contract_violations.jsonl`` can grow unbounded; we read
177:    the tail and re-pull more if the oldest entry is still inside the
178:    24h window.
179:
180:    Resolution-aware (added 2026-05-04). A row is treated as resolved when:
181:
182:    1. ``layer2_journal_activity`` — ``layer2_journal.jsonl`` has at least
183:       one entry with ``ts >= violation.details.trigger_time``. The
184:       journal entry IS the resolution; the contract check itself returns
185:       early on this condition the next cycle.
186:    2. The same incident is already represented by an *unresolved* row in
187:       ``critical_errors.jsonl`` (same ``invariant`` + per-invariant
188:       identity hash). The errors panel will surface that row; showing
189:       it again under "violations" would be cosmetic noise.
190:       *Resolved* critical_errors rows do NOT hide the violation — that
191:       hand-off must come through path 1 or path 3.
192:    3. ``critical_errors.jsonl`` has a resolution row whose
193:       ``resolves_ts`` matches the timestamp of a critical_errors row that
194:       in turn matches our violation's identity. (Production ``resolves_ts``
195:       points at the critical_errors row, not the contract_violations row,
196:       so the match must go via the critical_errors row's ``ts``.)
197:
198:    Cross-row dedup uses per-invariant identity hashing — the same
199:    ``_hash_violation_identity`` keys that ``loop_contract`` uses for
200:    Telegram cooldown — so distinct incidents with similar message text
201:    don't collapse into one row.
202:
203:    Without these filters the panel surfaced cleared-but-stale events for
204:    up to 24h after the underlying issue was fixed (observed 2026-05-04
205:    when 6 CV rows were shown despite Layer 2 working and accuracy
206:    regression already disposition'd by the 2026-05-03 research session).
207:    """
208:    try:
209:        entries = _load_last_n_hours(
210:            dd / "contract_violations.jsonl", hours=24, ts_field="ts"
211:        )
212:    except Exception as e:
213:        return {"unresolved": 0, "recent": [], "error": f"violations load: {type(e).__name__}: {e}"}
214:    try:
215:        crit_idx = _critical_errors_index(dd)
216:        latest_l2_journal_ts = _latest_layer2_journal_ts(dd)
217:
218:        # Pass 1: severity filter + per-row resolution check.
219:        kept: list[dict[str, Any]] = []
220:        for entry in entries:
221:            if not isinstance(entry, dict):
222:                continue
223:            if entry.get("severity") != "CRITICAL":
224:                continue
225:            if _violation_resolved(entry, crit_idx, latest_l2_journal_ts):
226:                continue
227:            kept.append(entry)
228:
229:        # Pass 2: cross-row dedup using per-invariant identity (mirrors
230:        # loop_contract._hash_violation_identity so two distinct incidents
231:        # that happen to share the same first 200 chars don't collapse
232:        # into one row, and two layer2_journal_activity violations on
233:        # different triggers stay separate even when the rendered text
234:        # rounds to the same minute count).
235:        seen: set[str] = set()
236:        deduped: list[dict[str, Any]] = []
237:        for entry in sorted(kept, key=lambda e: e.get("ts", ""), reverse=True):
238:            key = _violation_identity_key(entry)
239:            if key in seen:
240:                continue
241:            seen.add(key)
242:            deduped.append(entry)
243:
244:        recent = [
245:            {
246:                "ts": e.get("ts"),
247:                "invariant": e.get("invariant"),
248:                "severity": e.get("severity"),
249:                "message": (e.get("message") or "")[:200],
250:            }
251:            for e in deduped
252:        ]
253:        return {"unresolved": len(recent), "recent": recent[:5]}
254:    except Exception as e:
255:        return {"unresolved": 0, "recent": [], "error": f"violations aggregate: {type(e).__name__}: {e}"}
256:
257:
258:def _critical_errors_index(dd: Path) -> dict[str, Any]:
259:    """Index of critical_errors.jsonl for cross-stream resolution checks.
260:
261:    Returns a dict with:
262:
263:    - ``unresolved_keys``: set of ``(invariant, identity_key)`` tuples for
264:      *unresolved* critical-level entries. A contract_violations row whose
265:      identity matches one of these is already represented in the errors
266:      panel, so we hide it under violations to avoid double-counting.
267:    - ``resolved_keys``: set of ``(invariant, identity_key)`` tuples for
268:      critical-level entries that have been retroactively resolved (a
269:      later row pointed at them via ``resolves_ts``). Matching contract
270:      violations are treated as resolved.
271:
272:    Resolution rows themselves carry ``level == 'info'`` and a
273:    ``resolves_ts`` pointing at the original critical row's ``ts`` —
274:    same protocol as ``check_critical_errors.py``.
275:    """
276:    try:
277:        entries = load_jsonl(dd / "critical_errors.jsonl")
278:    except Exception:
279:        return {"unresolved_keys": set(), "resolved_keys": set()}
280:
281:    by_ts: dict[str, dict] = {}
282:    resolved_ts: set[str] = set()
283:    for entry in entries:
284:        if not isinstance(entry, dict):
285:            continue
286:        if entry.get("category") == "resolution" and entry.get("resolves_ts"):
287:            resolved_ts.add(str(entry["resolves_ts"]))
288:            continue
289:        if entry.get("level") and entry.get("level") != "critical":
290:            continue
291:        ts = entry.get("ts")
292:        if ts:
293:            by_ts[str(ts)] = entry
294:
295:    unresolved_keys: set[tuple[str, str]] = set()
296:    resolved_keys: set[tuple[str, str]] = set()
297:    for ts, entry in by_ts.items():
298:        # critical_errors rows record the invariant in either ``caller``
299:        # or ``category`` (the dispatcher uses the invariant name for
300:        # both). Prefer caller; fall back to category.
301:        invariant = (
302:            entry.get("caller") or entry.get("category") or ""
303:        )
304:        key = (str(invariant), _identity_key_for_dict(entry))
305:        if ts in resolved_ts:
306:            resolved_keys.add(key)
307:        else:
308:            unresolved_keys.add(key)
309:
310:    return {
311:        "unresolved_keys": unresolved_keys,
312:        "resolved_keys": resolved_keys,
313:    }
314:
315:
316:def _latest_layer2_journal_ts(dd: Path) -> str | None:
317:    """Newest ts from ``layer2_journal.jsonl``, or None if empty."""
318:    try:
319:        tail = load_jsonl_tail(dd / "layer2_journal.jsonl", max_entries=20)
320:    except Exception:
321:        return None
322:    best: str | None = None
323:    for e in tail:
324:        if not isinstance(e, dict):
325:            continue
326:        ts = e.get("ts") or e.get("timestamp")
327:        if ts and (best is None or str(ts) > best):
328:            best = str(ts)
329:    return best
330:
331:
332:def _violation_identity_key(entry: dict) -> str:
333:    """Per-invariant identity payload for cross-stream resolution checks.
334:
335:    Delegates to ``portfolio.loop_contract.violation_identity_payload`` —
336:    the *same* function the source uses for Telegram cooldown / dedup
337:    state — so the two sides cannot drift.
338:    """
339:    return _identity_key_for_dict(entry)
340:
341:
342:def _identity_key_for_dict(entry: dict) -> str:
343:    # contract_violations.jsonl uses ``invariant``, critical_errors.jsonl
344:    # uses ``caller`` (or ``category`` as fallback). The dispatcher writes
345:    # the invariant name into both fields for the periodic-violation path
346:    # (loop_contract:992-997), but the inline layer2 path writes
347:    # category="contract_violation" + caller=invariant_name
348:    # (loop_contract:471-476) — prefer caller, fall back to category.
349:    invariant = (
350:        entry.get("invariant")
351:        or entry.get("caller")
352:        or entry.get("category")
353:        or ""
354:    )
355:    raw_msg = entry.get("message") or ""
356:    # ViolationTracker promotes warnings by prepending
357:    # "ESCALATED (Nx consecutive): " — the source strips this before
358:    # hashing; mirror that here so escalated CV rows match their
359:    # pre-escalation form and their critical_errors counterpart.
360:    msg = _ESCALATED_PREFIX_RE.sub("", raw_msg, count=1)[:200]
361:
362:    # ``record_critical_error`` writes the payload under ``context``;
363:    # ``_log_violations`` writes it under ``details``. Accept either so
364:    # the cross-stream identity match works on both sides without a
365:    # wire-format change. (Claude review of a85a646f, P1-1.)
366:    payload_dict = entry.get("details") or entry.get("context") or {}
367:    if not isinstance(payload_dict, dict):
368:        payload_dict = {}
369:
370:    return violation_identity_payload(invariant, msg, payload_dict)
371:
372:
373:def _violation_resolved(
374:    entry: dict,
375:    crit_idx: dict[str, Any],
376:    latest_l2_journal_ts: str | None,
377:) -> bool:
378:    # Path 1: layer2_journal_activity is implicitly resolved by a later
379:    # journal entry — same condition the contract check itself uses next
380:    # cycle.
381:    if entry.get("invariant") == "layer2_journal_activity":
382:        details = entry.get("details") or {}
383:        trig = details.get("trigger_time")
384:        if trig and latest_l2_journal_ts and str(latest_l2_journal_ts) >= str(trig):
385:            return True
386:
387:    # Paths 2 + 3: cross-stream dedup via critical_errors.jsonl. A
388:    # contract violation that matches any unresolved or resolved
389:    # critical_errors entry is already accounted for: unresolved -> the
390:    # errors panel will surface it; resolved -> it's been cleared.
391:    key = (
392:        str(entry.get("invariant") or ""),
393:        _identity_key_for_dict(entry),
394:    )
395:    if key in crit_idx.get("resolved_keys", set()):
396:        return True
397:    if key in crit_idx.get("unresolved_keys", set()):
398:        return True
399:    return False
400:
401:
402:def _llm_inference(dd: Path) -> dict[str, Any]:
403:    """Per-LLM inference success rate.
404:
405:    Sources:
406:      * ``local_llm_report_latest.json -> health.{chronos,kronos}`` —
407:        forecast models, ``ok``/``fail`` counts over the report window.
408:      * ``health_state.json -> signal_health[*]`` — for signal-level LLM
409:        signals (``claude_fundamental``, ``forecast``). ``total_calls``
410:        and ``total_failures`` are lifetime counters maintained by
411:        ``portfolio.health.record_signal_call``.
412:
413:    Ministral and Qwen3 don't yet have inference-success counters in
414:    either source; their accuracy lives in ``local_llm_report_latest``
415:    but that is a different metric (was the BUY/SELL right at horizon).
416:    Kept out of this view rather than mislabelled.
417:
418:    Codex P2 follow-up: every numeric field is parsed defensively so
419:    a malformed ``{"ok": "oops"}`` row no longer crashes the whole
420:    payload — that model is skipped with no other side-effects.
421:    """
422:    try:
423:        health = load_json(dd / "health_state.json", default={}) or {}
424:        sh = health.get("signal_health", {}) if isinstance(health, dict) else {}
425:        if not isinstance(sh, dict):
426:            sh = {}
427:        llm_report = load_json(dd / "local_llm_report_latest.json", default={}) or {}
428:        llm_health = llm_report.get("health", {}) if isinstance(llm_report, dict) else {}
429:        if not isinstance(llm_health, dict):
430:            llm_health = {}
431:
432:        models: list[dict[str, Any]] = []
433:
434:        for key, label in (("chronos", "Chronos-2"), ("kronos", "Kronos")):
435:            h = llm_health.get(key)
436:            if not isinstance(h, dict):
437:                continue
438:            ok = _safe_int(h.get("ok"))
439:            fail = _safe_int(h.get("fail"))
440:            if ok is None or fail is None:
441:                continue
442:            total = ok + fail
443:            if total == 0:
444:                continue
445:            models.append(
446:                {
447:                    "name": label,
448:                    "key": key,
449:                    "total": total,
450:                    "failures": fail,
451:                    "success_pct": round(100.0 * ok / total, 1),
452:                }
453:            )
454:
455:        for key, label in (
456:            ("claude_fundamental", "Claude Fundamental"),
457:            ("forecast", "Forecast voter"),
458:        ):
459:            h = sh.get(key)
460:            if not isinstance(h, dict):
461:                continue
462:            total = _safe_int(h.get("total_calls"))
463:            fail = _safe_int(h.get("total_failures"))
464:            if total is None or fail is None or total <= 0:
465:                continue
466:            models.append(
467:                {
468:                    "name": label,
469:                    "key": key,
470:                    "total": total,
471:                    "failures": fail,
472:                    "success_pct": round(100.0 * (total - fail) / total, 1),
473:                    "last_failure_ts": h.get("last_failure"),
474:                }
475:            )
476:
477:        overall_pct: float | None = None
478:        if models:
479:            weight = sum(m["total"] for m in models)
480:            if weight > 0:
481:                overall_pct = round(
482:                    sum(m["success_pct"] * m["total"] for m in models) / weight, 1
483:                )
484:
485:        return {"models": models, "overall_pct": overall_pct}
486:    except Exception as e:
487:        return {"models": [], "overall_pct": None,
488:                "error": f"llm_inference: {type(e).__name__}: {e}"}
489:
490:
491:def _layer2_24h(dd: Path) -> dict[str, Any]:
492:    """Layer 2 trigger frequency + success rate over the last 24h.
493:
494:    A trigger fires whenever Layer 1 spawns a Claude CLI subprocess and
495:    appends to ``claude_invocations.jsonl``. ``status`` is one of
496:    ``invoked`` (success), ``timeout``, ``error``, ``blocked``.
497:
498:    Codex P2 follow-up: uses adaptive tail growth so a high-volume day
499:    (>2000 invocations in 24h) doesn't silently undercount.
500:    """
501:    try:
502:        entries = _load_last_n_hours(
503:            dd / "claude_invocations.jsonl", hours=24, ts_field="timestamp"
504:        )
505:    except Exception as e:
506:        return {
507:            "triggers_24h": 0,
508:            "success_24h": 0,
509:            "success_pct": None,
510:            "latest": None,
511:            "spark_24h": [0] * 24,
512:            "error": f"invocations load: {type(e).__name__}: {e}",
513:        }
514:    try:
515:        cutoff = datetime.now(UTC) - timedelta(hours=24)
516:        triggers: list[tuple[datetime, dict]] = []
517:        for entry in entries:
518:            if not isinstance(entry, dict):
519:                continue
520:            ts_raw = entry.get("timestamp")
521:            if not ts_raw:
522:                continue
523:            try:
524:                ts = _parse_ts(str(ts_raw))
525:            except Exception:
526:                continue
527:            if ts < cutoff:
528:                continue
529:            triggers.append((ts, entry))
530:
531:        triggers.sort(key=lambda x: x[0])
532:        success = sum(1 for _, e in triggers if e.get("status") == "invoked")
533:        pct = round(100.0 * success / len(triggers), 1) if triggers else None
534:        latest_entry = triggers[-1][1] if triggers else None
535:
536:        now = datetime.now(UTC)
537:        buckets = [0] * 24
538:        for ts, _ in triggers:
539:            hours_ago = int((now - ts).total_seconds() // 3600)
540:            if 0 <= hours_ago < 24:
541:                buckets[23 - hours_ago] += 1
542:
543:        latest_payload: dict[str, Any] | None = None
544:        if latest_entry is not None:
545:            latest_payload = {
546:                "ts": latest_entry.get("timestamp"),
547:                "caller": latest_entry.get("caller"),
548:                "status": latest_entry.get("status"),
549:                "duration_seconds": latest_entry.get("duration_seconds"),
550:                "model": latest_entry.get("model"),
551:            }
552:
553:        return {
554:            "triggers_24h": len(triggers),
555:            "success_24h": success,
556:            "success_pct": pct,
557:            "latest": latest_payload,
558:            "spark_24h": buckets,
559:        }
560:    except Exception as e:
561:        return {
562:            "triggers_24h": 0,
563:            "success_24h": 0,
564:            "success_pct": None,
565:            "latest": None,
566:            "spark_24h": [0] * 24,
567:            "error": f"layer2 aggregate: {type(e).__name__}: {e}",
568:        }
569:
570:
571:def _signal_aggregate(dd: Path) -> dict[str, Any]:
572:    """Latest signal_log entry collapsed into per-ticker counts.
573:
574:    ``total_voters`` in the source is BUY+SELL only (active voters);
575:    every other signal in the dict counts as HOLD/abstain. We expose
576:    both ``hold`` (literal vote count) and ``abstain`` (alias) so the
577:    UI can word the row either way.
578:
579:    Codex P2 follow-up: tolerate the latest entry being a non-dict
580:    (e.g. an empty list or null) — the section reports an in-band
581:    error instead of bubbling AttributeError up to the route.
582:    """
583:    try:
584:        entries = load_jsonl_tail(dd / "signal_log.jsonl", max_entries=5)
585:    except Exception as e:
586:        return {"tickers": [], "error": f"signal_log load: {type(e).__name__}: {e}"}
587:    if not entries:
588:        return {"tickers": []}
589:    last = entries[-1]
590:    if not isinstance(last, dict):
591:        return {"tickers": [], "error": "signal_log: last entry is not a JSON object"}
592:    try:
593:        tickers_dict = last.get("tickers", {}) or {}
594:        if not isinstance(tickers_dict, dict):
595:            return {"ts": last.get("ts"), "tickers": [],
596:                    "error": "signal_log.tickers is not a JSON object"}
597:        tickers: list[dict[str, Any]] = []
598:        for sym, data in tickers_dict.items():
599:            if not isinstance(data, dict):
600:                continue
601:            signals = data.get("signals", {}) or {}
602:            if not isinstance(signals, dict):
603:                continue
604:            total = len(signals)
605:            buy = sum(1 for v in signals.values() if v == "BUY")
606:            sell = sum(1 for v in signals.values() if v == "SELL")
607:            hold = total - buy - sell
608:            active = buy + sell
609:            confidence = (active / total) if total else 0.0
610:            tickers.append(
611:                {
612:                    "ticker": sym,
613:                    "consensus": data.get("consensus", "HOLD"),
614:                    "buy": buy,
615:                    "sell": sell,
616:                    "hold": hold,
617:                    "abstain": hold,
618:                    "total": total,
619:                    "confidence": round(confidence, 3),
620:                    "regime": data.get("regime"),
621:                }
622:            )
623:        return {"ts": last.get("ts"), "tickers": tickers}
624:    except Exception as e:
625:        return {"tickers": [], "error": f"signal_aggregate: {type(e).__name__}: {e}"}
626:
627:
628:def _pnl_footer(dd: Path) -> dict[str, Any]:
629:    """Single-line P&L for the deprioritised portfolio footer."""
630:    try:
631:        ps = load_json(dd / "portfolio_state.json", default={}) or {}
632:        pb = load_json(dd / "portfolio_state_bold.json", default={}) or {}
633:        if not isinstance(ps, dict):
634:            ps = {}
635:        if not isinstance(pb, dict):
636:            pb = {}
637:        return {
638:            "patient_value_sek": ps.get("portfolio_value", ps.get("equity_sek")),
639:            "bold_value_sek": pb.get("portfolio_value", pb.get("equity_sek")),
640:            "patient_starting_sek": ps.get("starting_capital", 500_000.0),
641:            "bold_starting_sek": pb.get("starting_capital", 500_000.0),
642:        }
643:    except Exception as e:
644:        return {"error": f"pnl load: {type(e).__name__}: {e}"}
645:
646:
647:# ---------------------------------------------------------------------------
648:# Severity rollup
649:# ---------------------------------------------------------------------------
650:
651:
652:def _color(payload: dict[str, Any]) -> tuple[str, list[str]]:
653:    """Compute overall GREEN/YELLOW/RED + a list of reasons.
654:
655:    YELLOW means "look at this when you next glance at the dashboard."
656:    RED means "something is actually broken right now."
657:    """
658:    severity = "GREEN"
659:    reasons: list[str] = []
660:
661:    def bump(level: str) -> None:
662:        nonlocal severity
663:        rank = {"GREEN": 0, "YELLOW": 1, "RED": 2}
664:        if rank[level] > rank[severity]:
665:            severity = level
666:
667:    hb = payload.get("heartbeat") or {}
668:    age = hb.get("age_seconds")
669:    if age is None:
670:        bump("RED")
671:        reasons.append("loop heartbeat: unknown")
672:    elif age > HEARTBEAT_YELLOW_S:
673:        bump("RED")
674:        reasons.append(f"loop silent {int(age / 60)}m")
675:    elif age > HEARTBEAT_GREEN_S:
676:        bump("YELLOW")
677:        reasons.append(f"loop heartbeat {int(age)}s ago")
678:
679:    err = payload.get("errors") or {}
680:    n = err.get("unresolved", 0) or 0
681:    if n > ERRORS_YELLOW_MAX:
682:        bump("RED")
683:        reasons.append(f"{n} unresolved errors")
684:    elif n > 0:
685:        bump("YELLOW")
686:        reasons.append(f"{n} unresolved error{'s' if n != 1 else ''}")
687:
688:    cv = payload.get("contract_violations") or {}
689:    vn = cv.get("unresolved", 0) or 0
690:    if vn > 5:
691:        bump("RED")
692:        reasons.append(f"{vn} contract violations 24h")
693:    elif vn > 0:
694:        bump("YELLOW")
695:        reasons.append(f"{vn} contract violation{'s' if vn != 1 else ''} 24h")
696:
697:    llm = payload.get("llm_inference") or {}
698:    pct = llm.get("overall_pct")
699:    if pct is not None:
700:        if pct < LLM_YELLOW_PCT:
701:            bump("RED")
702:            reasons.append(f"LLM inference {pct}%")
703:        elif pct < LLM_GREEN_PCT:
704:            bump("YELLOW")
705:            reasons.append(f"LLM inference {pct}%")
706:
707:    l2 = payload.get("layer2") or {}
708:    l2pct = l2.get("success_pct")
709:    if (
710:        l2pct is not None
711:        and (l2.get("triggers_24h", 0) or 0) >= LAYER2_MIN_TRIGGERS_FOR_GATE
712:    ):
713:        if l2pct < LAYER2_YELLOW_PCT:
714:            bump("RED")
715:            reasons.append(f"Layer 2 success {l2pct}%")
716:        elif l2pct < LAYER2_GREEN_PCT:
717:            bump("YELLOW")
718:            reasons.append(f"Layer 2 success {l2pct}%")
719:
720:    if not reasons:
721:        reasons = ["all systems nominal"]
722:    return severity, reasons
723:
724:
725:# ---------------------------------------------------------------------------
726:# Helpers
727:# ---------------------------------------------------------------------------
728:
729:
730:def _parse_ts(s: str) -> datetime:
731:    """Parse an ISO timestamp tolerating trailing Z."""
732:    s = s.replace("Z", "+00:00")
733:    return datetime.fromisoformat(s)
734:
735:
736:def _safe_int(v: Any) -> int | None:
737:    """``int(v)`` that returns None instead of raising on bad input.
738:
739:    Used so a ``{"ok": "oops"}`` row inside local_llm_report skips the
740:    affected model rather than 500'ing the whole endpoint (codex P2
741:    finding 2026-05-04).
742:    """
743:    try:
744:        return int(v)
745:    except (TypeError, ValueError):
746:        return None
747:
748:
749:def _load_last_n_hours(path: Path, *, hours: int, ts_field: str) -> list[dict]:
750:    """Tail-read a jsonl file growing the window until the oldest entry
751:    is older than the cutoff or we've fully scanned the file.
752:
753:    Used by ``_layer2_24h`` and ``_violations_recent`` to make the
754:    "last 24h" claim authoritative even when activity spikes (codex
755:    P2 finding 2026-05-04: previous tail of 2000 lines silently
756:    undercounted on high-volume days).
757:
758:    The returned list is in file order (oldest first). Every step
759:    doubles the tail size up to a 50K cap; if the cap is hit we fall
760:    back to a full ``load_jsonl`` so the count remains correct.
761:    """
762:    cutoff = datetime.now(UTC) - timedelta(hours=hours)
763:    cutoff_iso = cutoff.isoformat()
764:    for max_entries in (500, 2_000, 10_000, 50_000):
765:        rows = load_jsonl_tail(
766:            path, max_entries=max_entries, tail_bytes=max_entries * 512
767:        )
768:        if not rows:
769:            return []
770:        oldest = next(
771:            (r.get(ts_field) for r in rows if isinstance(r, dict) and r.get(ts_field)),
772:            None,
773:        )
774:        if not oldest or str(oldest) < cutoff_iso:
775:            return [r for r in rows if isinstance(r, dict)
776:                    and (r.get(ts_field) or "") >= cutoff_iso]
777:    rows = load_jsonl(path)
778:    return [r for r in rows if isinstance(r, dict)
779:            and (r.get(ts_field) or "") >= cutoff_iso]
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 1130ms:
1:"""DEPRECATED 2026-04-30 — exported snapshots are no longer served publicly.
2:
3:Originally, this script wrote each API endpoint's JSON to
4:dashboard/static/api-data/<name>.json so the GitHub Pages clone of the
5:dashboard could read snapshots when no live backend was available. The
6:public dashboard now lives at https://raanman.lol behind a Cloudflare
7:tunnel + cookie auth (see docs/TUNNEL_SETUP.md), so the snapshot files
8:have no legitimate public consumer — and any *.json under static/ is
9:reachable at /static/api-data/<name>.json with NO auth (Flask's default
10:static handler), which is a leak.
11:
12:The directory dashboard/static/api-data/ is now in .gitignore. Running
13:this script locally still produces files there but they are not committed
14:and (more importantly) the live deployment no longer has any code path
15:that reads them. Kept on disk only because tests/test_dashboard_export_static.py
16:imports it.
17:
18:If you need a static export for some other purpose, write to a directory
19:*outside* dashboard/static/ (e.g. dashboard/_export/) so Flask doesn't
20:auto-serve it.
21:"""
22:
23:import json
24:import sys
25:from pathlib import Path
26:
27:# Ensure project root is importable
28:PROJECT_ROOT = Path(__file__).resolve().parent.parent
29:sys.path.insert(0, str(PROJECT_ROOT))
30:CONFIG_PATH = PROJECT_ROOT / "config.json"
31:
32:from dashboard.app import _json_safe, app  # noqa: E402
33:
34:# Endpoints to export: (Flask route, static filename)
35:ENDPOINTS = [
36:    ("/api/summary", "summary.json"),
37:    ("/api/signals", "signals.json"),
38:    ("/api/portfolio", "portfolio.json"),
39:    ("/api/portfolio-bold", "portfolio-bold.json"),
40:    ("/api/accuracy", "accuracy.json"),
41:    ("/api/signal-heatmap", "signal-heatmap.json"),
42:    ("/api/equity-curve", "equity-curve.json"),
43:    ("/api/triggers", "triggers.json"),
44:    ("/api/decisions", "decisions.json"),
45:    ("/api/telegrams", "telegrams.json"),
46:    ("/api/accuracy-history", "accuracy-history.json"),
47:    ("/api/local-llm-trends", "local-llm-trends.json"),
48:    ("/api/trades", "trades.json"),
49:    ("/api/warrants", "warrants.json"),
50:    ("/api/risk", "risk.json"),
51:    ("/api/metals", "metals.json"),
52:    ("/api/golddigger", "golddigger.json"),
53:    ("/api/metals-accuracy", "metals-accuracy.json"),
54:    ("/api/lora-status", "lora-status.json"),
55:    ("/api/health", "health.json"),
56:]
57:
58:DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "static" / "api-data"
59:
60:
61:def _get_dashboard_token():
62:    """Read dashboard token from config.json, if configured."""
63:    try:
64:        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
65:    except (OSError, json.JSONDecodeError):
66:        return None
67:    return cfg.get("dashboard_token") or None
68:
69:
70:def export_all(out_dir: Path | None = None) -> dict:
71:    """Export all API endpoints to static JSON files.
72:
73:    Args:
74:        out_dir: Directory to write files to. Defaults to dashboard/static/api-data/.
75:
76:    Returns:
77:        Dict with 'ok' (list of exported filenames) and 'failed' (list of dicts).
78:    """
79:    if out_dir is None:
80:        out_dir = DEFAULT_OUT_DIR
81:    out_dir.mkdir(parents=True, exist_ok=True)
82:    token = _get_dashboard_token()
83:
84:    ok = []
85:    failed = []
86:
87:    with app.test_client() as client:
88:        for route, filename in ENDPOINTS:
89:            try:
90:                req_route = route
91:                if token:
92:                    sep = "&" if "?" in req_route else "?"
93:                    req_route = f"{req_route}{sep}token={token}"
94:                resp = client.get(req_route)
95:                if resp.status_code != 200:
96:                    failed.append({
97:                        "route": route,
98:                        "filename": filename,
99:                        "status": resp.status_code,
100:                        "error": resp.get_data(as_text=True)[:200],
101:                    })
102:                    continue
103:
104:                data = resp.get_json()
105:                dest = out_dir / filename
106:                with open(dest, "w", encoding="utf-8") as f:
107:                    json.dump(
108:                        _json_safe(data),
109:                        f,
110:                        ensure_ascii=False,
111:                        separators=(",", ":"),
112:                        allow_nan=False,
113:                    )
114:
115:                ok.append(filename)
116:            except Exception as e:
117:                failed.append({
118:                    "route": route,
119:                    "filename": filename,
120:                    "error": str(e),
121:                })
122:
123:    return {"ok": ok, "failed": failed}
124:
125:
126:def main():
127:    import argparse
128:
129:    parser = argparse.ArgumentParser(description="Export dashboard API to static JSON")
130:    parser.add_argument(
131:        "--out",
132:        type=Path,
133:        default=None,
134:        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
135:    )
136:    args = parser.parse_args()
137:
138:    result = export_all(out_dir=args.out)
139:
140:    print(f"Exported {len(result['ok'])}/{len(ENDPOINTS)} endpoints")
141:    for name in result["ok"]:
142:        print(f"  OK  {name}")
143:    for fail in result["failed"]:
144:        print(f"  FAIL {fail['filename']}: {fail.get('status', '')} {fail.get('error', '')}")
145:
146:    if result["failed"]:
147:        sys.exit(1)
148:
149:
150:if __name__ == "__main__":
151:    main()
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 1171ms:
1:"""House-hunting dashboard — read-only viewer at /house over the
2:househunting project's findapartments runs and innerstad heatmap.
3:
4:Mounted as a Flask Blueprint on the main dashboard app. Reuses the same
5:`require_auth` cookie/token gate, so the entire `/house/*` surface is
6:protected by the same `pf_dashboard_token` that gates the finance dashboard.
7:
8:Reads files directly from disk under `<house_root>`:
9:  - <house_root>/data/findapartments/<run-id>/_manifest.json
10:  - <house_root>/data/findapartments/<run-id>/_summary.thesis.md  (preferred)
11:                                            /_summary.md           (fallback)
12:  - <house_root>/data/findapartments/<run-id>/<slug>.thesis.md     (preferred)
13:                                            /<slug>.md             (fallback)
14:  - <house_root>/data/findapartments/<run-id>/_raw/<slug>/data.json
15:  - <house_root>/output/heatmap.html
16:
17:`house_root` is configured via `config.json[house_root]`, defaulting to
18:`Q:\\househunting`. The blueprint never imports from the househunting
19:project — it's a pure file viewer, so the two repos stay decoupled.
20:
21:SECURITY: Every route is wrapped with `@require_auth`. There's a unit
22:test (`test_dashboard_house.py::test_every_route_requires_auth`) that
23:walks the blueprint's URL map and asserts each route returns 401 without
24:a cookie. Asset files are streamed through authenticated routes —
25:NEVER copied or symlinked into `dashboard/static/` (that path is served
26:by Flask's default static handler without auth, per docs/TUNNEL_SETUP.md).
27:"""
28:from __future__ import annotations
29:
30:import json
31:import re
32:from datetime import datetime
33:from html import escape
34:from pathlib import Path
35:from typing import Optional
36:
37:from flask import (
38:    Blueprint, abort, jsonify, redirect, request, send_file,
39:)
40:from werkzeug.utils import secure_filename
41:
42:import markdown as md_lib  # type: ignore[import-not-found]
43:
44:from dashboard.auth import _get_config, require_auth
45:
46:bp = Blueprint("house", __name__, url_prefix="/house")
47:
48:
49:# ---------------------------------------------------------------------------
50:# Config + path helpers
51:# ---------------------------------------------------------------------------
52:
53:
54:def _house_root() -> Path:
55:    """Root of the house-hunting project on disk. Configurable via
56:    `config.json[house_root]`. Defaults to Q:\\househunting (the local
57:    canonical path on the dashboard host)."""
58:    cfg = _get_config()
59:    return Path(cfg.get("house_root", r"Q:\househunting"))
60:
61:
62:def _runs_dir() -> Path:
63:    return _house_root() / "data" / "findapartments"
64:
65:
66:def _heatmap_path() -> Path:
67:    return _house_root() / "output" / "heatmap.html"
68:
69:
70:# Run IDs look like 2026-05-01-0032 — YYYY-MM-DD-HHMM. Slugs look like
71:# lagenhet-3rum-kungsholmen-... — lowercase ASCII + digits + hyphens.
72:# Both are URL-safe but we still validate before using as a path component.
73:_RUN_ID_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}(?:-[0-9]{4})?$")
74:_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,200}$")
75:
76:
77:def _validate_run_id(run_id: str) -> str:
78:    if not _RUN_ID_RE.match(run_id):
79:        abort(404)
80:    return run_id
81:
82:
83:def _validate_slug(slug: str) -> str:
84:    # secure_filename strips path traversal; the regex enforces shape.
85:    cleaned = secure_filename(slug)
86:    if cleaned != slug or not _SLUG_RE.match(cleaned):
87:        abort(404)
88:    return cleaned
89:
90:
91:# ---------------------------------------------------------------------------
92:# Run discovery
93:# ---------------------------------------------------------------------------
94:
95:
96:def _list_runs() -> list[dict]:
97:    """List all run directories newest-first. Returns empty list if the
98:    findapartments dir doesn't exist (fresh install, etc.)."""
99:    runs_dir = _runs_dir()
100:    if not runs_dir.exists():
101:        return []
102:    runs: list[dict] = []
103:    for entry in runs_dir.iterdir():
104:        if not entry.is_dir() or not _RUN_ID_RE.match(entry.name):
105:            continue
106:        manifest = entry / "_manifest.json"
107:        try:
108:            slugs = json.loads(manifest.read_text())
109:        except (FileNotFoundError, json.JSONDecodeError):
110:            slugs = []
111:        runs.append({
112:            "run_id": entry.name,
113:            "candidate_count": len(slugs),
114:            "has_summary": (entry / "_summary.thesis.md").exists()
115:                           or (entry / "_summary.md").exists(),
116:            "modified_iso": datetime.fromtimestamp(
117:                entry.stat().st_mtime
118:            ).isoformat(timespec="seconds"),
119:        })
120:    runs.sort(key=lambda r: r["run_id"], reverse=True)
121:    return runs
122:
123:
124:def _resolve_md(run_id: str, basename: str) -> Optional[Path]:
125:    """Resolve <run-id>/<basename>.thesis.md, falling back to .md."""
126:    base = _runs_dir() / run_id
127:    thesis = base / f"{basename}.thesis.md"
128:    if thesis.exists():
129:        return thesis
130:    legacy = base / f"{basename}.md"
131:    if legacy.exists():
132:        return legacy
133:    return None
134:
135:
136:# ---------------------------------------------------------------------------
137:# HTML shell — minimal, mobile-friendly
138:# ---------------------------------------------------------------------------
139:
140:
141:_PAGE_CSS = """
142::root { color-scheme: light dark; }
143:* { box-sizing: border-box; }
144:html, body { margin: 0; padding: 0; }
145:body {
146:  font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI",
147:        system-ui, sans-serif;
148:  max-width: 900px; margin: 0 auto; padding: 16px 20px 80px;
149:  color: #222; background: #fff;
150:}
151:@media (prefers-color-scheme: dark) {
152:  body { color: #e8e8e8; background: #111; }
153:  a { color: #6cf; }
154:  a:visited { color: #c9f; }
155:  table { border-color: #333; }
156:  th, td { border-color: #333; }
157:  thead th { background: #1a1a1a; }
158:  code { background: #1a1a1a; }
159:}
160:nav.crumbs { font-size: 14px; padding: 8px 0; border-bottom: 1px solid #ddd;
161:             margin-bottom: 16px; }
162:nav.crumbs a { margin-right: 8px; }
163:h1 { font-size: 26px; line-height: 1.25; margin-top: 8px; }
164:h2 { font-size: 20px; margin-top: 28px; padding-bottom: 4px;
165:     border-bottom: 1px solid #eee; }
166:h3 { font-size: 17px; }
167:table { border-collapse: collapse; width: 100%; margin: 12px 0; }
168:th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left;
169:         vertical-align: top; font-size: 14px; }
170:thead th { background: #f4f4f4; }
171:code, pre { font-family: ui-monospace, "Menlo", "Consolas", monospace;
172:            font-size: 13px; }
173:pre { background: #f6f8fa; padding: 10px; overflow-x: auto;
174:      border-radius: 4px; }
175:code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }
176:blockquote { border-left: 3px solid #88c; padding: 4px 12px;
177:             margin: 12px 0; color: #444; }
178:@media (prefers-color-scheme: dark) {
179:  blockquote { color: #aaa; }
180:}
181:.runs-list { list-style: none; padding: 0; }
182:.runs-list li { padding: 10px 12px; border: 1px solid #ddd;
183:                border-radius: 4px; margin-bottom: 8px; }
184:.runs-list .meta { color: #666; font-size: 13px; }
185:@media (prefers-color-scheme: dark) {
186:  .runs-list li { border-color: #333; }
187:  .runs-list .meta { color: #aaa; }
188:}
189:"""
190:
191:
192:def _shell(title: str, body_html: str, breadcrumbs: list[tuple[str, str]]) -> str:
193:    crumbs_html = " ›\n".join(
194:        f'<a href="{escape(href)}">{escape(label)}</a>'
195:        for href, label in breadcrumbs
196:    )
197:    return (
198:        f"<!doctype html>\n"
199:        f"<html lang=\"en\"><head>\n"
200:        f"<meta charset=\"utf-8\">\n"
201:        f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
202:        f"<title>{escape(title)}</title>\n"
203:        f"<style>{_PAGE_CSS}</style>\n"
204:        f"</head><body>\n"
205:        f"<nav class=\"crumbs\">{crumbs_html}</nav>\n"
206:        f"{body_html}\n"
207:        f"</body></html>\n"
208:    )
209:
210:
211:def _render_markdown(text: str) -> str:
212:    return md_lib.markdown(
213:        text,
214:        extensions=["tables", "fenced_code", "sane_lists"],
215:        output_format="html5",
216:    )
217:
218:
219:# ---------------------------------------------------------------------------
220:# Routes — HTML
221:# ---------------------------------------------------------------------------
222:
223:
224:@bp.route("/")
225:@require_auth
226:def index():
227:    """Land on the most recent run; if no runs exist, show the empty list."""
228:    runs = _list_runs()
229:    if request.args.get("token"):
230:        # User auth-bootstrapped via ?token= — strip it from the URL.
231:        return redirect("/house", code=302)
232:    if not runs:
233:        body = (
234:            "<h1>House</h1>"
235:            "<p>No findapartments runs yet. From "
236:            "<code>Q:\\househunting</code>, run "
237:            "<code>.venv\\Scripts\\python -m scripts.findapartments_scan</code>.</p>"
238:        )
239:        return _shell("House — no runs", body, [("/house", "House")])
240:    return redirect(f"/house/runs/{runs[0]['run_id']}", code=302)
241:
242:
243:@bp.route("/runs")
244:@require_auth
245:def runs_list():
246:    runs = _list_runs()
247:    if not runs:
248:        body = "<h1>House</h1><p>No findapartments runs yet.</p>"
249:    else:
250:        items = []
251:        for r in runs:
252:            items.append(
253:                f"<li>"
254:                f"<a href=\"/house/runs/{escape(r['run_id'])}\">"
255:                f"{escape(r['run_id'])}</a>"
256:                f"<div class=\"meta\">{r['candidate_count']} candidates · "
257:                f"updated {escape(r['modified_iso'])}"
258:                f"{' · summary available' if r['has_summary'] else ''}"
259:                f"</div></li>"
260:            )
261:        body = (
262:            f"<h1>House — {len(runs)} run(s)</h1>"
263:            f"<ul class=\"runs-list\">{''.join(items)}</ul>"
264:            f"<p><a href=\"/house/heatmap\">Innerstad appreciation heatmap →</a></p>"
265:        )
266:    return _shell(
267:        "House — runs",
268:        body,
269:        [("/", "Dashboard"), ("/house", "House"), ("/house/runs", "Runs")],
270:    )
271:
272:
273:@bp.route("/runs/<run_id>")
274:@require_auth
275:def run_detail(run_id: str):
276:    run_id = _validate_run_id(run_id)
277:    summary = _resolve_md(run_id, "_summary")
278:    if not summary:
279:        abort(404)
280:    text = summary.read_text(encoding="utf-8")
281:    # Rewrite plain-text references to candidate slugs into hyperlinks. The
282:    # summary's table column is `<address>` text — slugs themselves aren't
283:    # in the rendered table but ARE in the per-candidate report file names.
284:    # Add a "candidates" footer with explicit links sourced from the manifest.
285:    manifest = _runs_dir() / run_id / "_manifest.json"
286:    candidate_links = ""
287:    try:
288:        slugs = json.loads(manifest.read_text())
289:    except (FileNotFoundError, json.JSONDecodeError):
290:        slugs = []
291:    if slugs:
292:        link_items = "".join(
293:            f"<li><a href=\"/house/runs/{escape(run_id)}/{escape(s)}\">"
294:            f"{escape(s)}</a></li>"
295:            for s in slugs
296:        )
297:        candidate_links = (
298:            f"<h2>All candidates ({len(slugs)})</h2>"
299:            f"<ul>{link_items}</ul>"
300:            f"<p><a href=\"/house/runs/{escape(run_id)}/_manifest.json\">"
301:            f"manifest.json</a> · "
302:            f"<a href=\"/house/heatmap\">heatmap</a></p>"
303:        )
304:    body = _render_markdown(text) + candidate_links
305:    return _shell(
306:        f"House — {run_id}",
307:        body,
308:        [("/", "Dashboard"), ("/house/runs", "Runs"), (f"/house/runs/{run_id}", run_id)],
309:    )
310:
311:
312:@bp.route("/runs/<run_id>/_manifest.json")
313:@require_auth
314:def run_manifest(run_id: str):
315:    run_id = _validate_run_id(run_id)
316:    manifest = _runs_dir() / run_id / "_manifest.json"
317:    if not manifest.exists():
318:        abort(404)
319:    return send_file(manifest, mimetype="application/json")
320:
321:
322:@bp.route("/runs/<run_id>/<slug>")
323:@require_auth
324:def candidate_detail(run_id: str, slug: str):
325:    run_id = _validate_run_id(run_id)
326:    slug = _validate_slug(slug)
327:    md_path = _resolve_md(run_id, slug)
328:    if not md_path:
329:        abort(404)
330:    text = md_path.read_text(encoding="utf-8")
331:    body = (
332:        _render_markdown(text)
333:        + f"<p><a href=\"/house/runs/{escape(run_id)}/{escape(slug)}/raw\">"
334:          "raw data.json →</a></p>"
335:    )
336:    return _shell(
337:        f"House — {slug}",
338:        body,
339:        [
340:            ("/", "Dashboard"),
341:            ("/house/runs", "Runs"),
342:            (f"/house/runs/{run_id}", run_id),
343:            (f"/house/runs/{run_id}/{slug}", slug),
344:        ],
345:    )
346:
347:
348:@bp.route("/runs/<run_id>/<slug>/raw")
349:@require_auth
350:def candidate_raw(run_id: str, slug: str):
351:    run_id = _validate_run_id(run_id)
352:    slug = _validate_slug(slug)
353:    raw = _runs_dir() / run_id / "_raw" / slug / "data.json"
354:    if not raw.exists():
355:        abort(404)
356:    return send_file(raw, mimetype="application/json")
357:
358:
359:@bp.route("/heatmap")
360:@require_auth
361:def heatmap():
362:    path = _heatmap_path()
363:    if not path.exists():
364:        abort(404)
365:    return send_file(path, mimetype="text/html")
366:
367:
368:# ---------------------------------------------------------------------------
369:# Routes — JSON API (for programmatic / phone clients)
370:# ---------------------------------------------------------------------------
371:
372:
373:@bp.route("/api/runs")
374:@require_auth
375:def api_runs():
376:    return jsonify({"runs": _list_runs()})
377:
378:
379:@bp.route("/api/runs/<run_id>")
380:@require_auth
381:def api_run(run_id: str):
382:    run_id = _validate_run_id(run_id)
383:    manifest = _runs_dir() / run_id / "_manifest.json"
384:    if not manifest.exists():
385:        abort(404)
386:    try:
387:        slugs = json.loads(manifest.read_text())
388:    except json.JSONDecodeError:
389:        abort(500)
390:    return jsonify({"run_id": run_id, "candidates": slugs})
391:
392:
393:@bp.route("/api/runs/<run_id>/<slug>")
394:@require_auth
395:def api_candidate(run_id: str, slug: str):
396:    run_id = _validate_run_id(run_id)
397:    slug = _validate_slug(slug)
398:    raw = _runs_dir() / run_id / "_raw" / slug / "data.json"
399:    if not raw.exists():
400:        abort(404)
401:    return send_file(raw, mimetype="application/json")
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "''^" dashboard/trading_status.py' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"@app\\.route|@bp\\.route|methods=|def api_|def house|def .*\\(\" dashboard/app.py dashboard/house_blueprint.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"json\\.loads\\(|json\\.load\\(|read_text\\(|open\\(|write_text\\(|os\\.replace|send_file\\(|redirect\\(|set_cookie|Cf-Access|Authorization|token=|force=|POST|GET|delete|remove|rename|unlink|mkdir\" dashboard/app.py dashboard/auth.py dashboard/export_static.py dashboard/house_blueprint.py dashboard/system_status.py dashboard/trading_status.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"json\\.loads\\(|json\\.load\\(|read_text\\(|open\\(|write_text\\(|atomic_write_json|atomic_append_jsonl|load_json|load_jsonl|load_jsonl_tail|subprocess\\.|shell=True|Popen\\(|run\\(|CreateJobObject|AssignProcessToJobObject|taskkill|wmic|powershell|ExecutionPolicy|CLAUDECODE|fix_agent.disabled|Read,Edit,Bash|claude -p|claude.exe|cooldown|backoff|disabled|set CLAUDECODE|schtasks|lock|singleton\" portfolio/file_utils.py portfolio/process_lock.py portfolio/subprocess_utils.py portfolio/config_validator.py portfolio/message_store.py portfolio/shadow_registry.py portfolio/vector_memory.py portfolio/backtester.py scripts/check_critical_errors.py scripts/fix_agent_dispatcher.py scripts/win/*.bat scripts/win/*.ps1 conftest.py" in Q:\finance-analyzer
 succeeded in 726ms:
1:"""Per-bot trading-status reader for the dashboard home page.
2:
3:Every Avanza-trading bot writes a state file. This module reads them
4:and projects each onto a small UI-shaped dict:
5:
6:    {"bot": "golddigger",
7:     "label": "GoldDigger",
8:     "state": "SCANNING" | "TRADING" | "HALTED" | "COOLDOWN"
9:              | "OUTSIDE_HOURS" | "UNKNOWN",
10:     "reason": "<human-readable why>",
11:     "since_seconds": <int|None>,
12:     "position": {...} | None,
13:     "stats": {...},
14:     "error": "<optional>"}
15:
16:The user's question "why aren't the loops trading on Avanza?" is
17:answered by the (state, reason) pair. State precedence:
18:
19:    1. UNKNOWN  — state file missing or unreadable
20:    2. HALTED   — bot wrote halted=True (reads halted_reason)
21:    3. COOLDOWN — fishing engine: last_trade + cooldown_s > now
22:    4. TRADING  — open position present
23:    5. OUTSIDE_HOURS — outside the Avanza trading window
24:                      (08:30–21:30 Europe/Stockholm)
25:    6. SCANNING — running normally, no signal strong enough yet
26:
27:GoldDigger and Elongir maintain ``halted_reason`` themselves, so we
28:surface it verbatim. Metals + fishing don't yet persist a "why no
29:trade" reason, so we fall back to inference from their state. See
30:``docs/PLAN.md`` and the plan file at
31:``/root/.claude/plans/merry-tinkering-cake.md``.
32:"""
33:
34:from __future__ import annotations
35:
36:from datetime import datetime, time as dtime, timezone
37:from pathlib import Path
38:from typing import Any
39:from zoneinfo import ZoneInfo
40:
41:from portfolio.file_utils import load_json
42:
43:DATA_DIR = Path(__file__).resolve().parent.parent / "data"
44:
45:# Avanza trading session in Europe/Stockholm — DST handled by zoneinfo.
46:# 2026-05-11: unified to 08:30–21:30 across all four bots after a user
47:# report that the dashboard was rendering OUTSIDE_HOURS at 14:23 CEST
48:# even though Elongir's actual config session is 08:30–21:30. The old
49:# 15:30–21:55 window matched GoldDigger's US-focused config and was
50:# misapplied here to metals/elongir/fishing. EU open is ~09:00 CET, US
51:# close is ~22:00 CET; 08:30–21:30 brackets the warrant-tradeable window
52:# the user actually trades on and matches the per-bot configs after the
53:# parallel patch to portfolio/golddigger/config.py.
54:SESSION_TZ = ZoneInfo("Europe/Stockholm")
55:SESSION_OPEN = dtime(8, 30)
56:SESSION_CLOSE = dtime(21, 30)
57:
58:UTC = timezone.utc
59:
60:
61:def compute(
62:    data_dir: Path | None = None,
63:    now_utc: datetime | None = None,
64:) -> dict[str, Any]:
65:    """Build the trading-status payload for all four Avanza bots."""
66:    dd = Path(data_dir) if data_dir else DATA_DIR
67:    now = now_utc or datetime.now(UTC)
68:    return {
69:        "ts": now.isoformat(),
70:        "session_open": _in_session(now),
71:        "bots": [
72:            _golddigger(dd, now),
73:            _elongir(dd, now),
74:            _metals(dd, now),
75:            _fishing(dd, now),
76:        ],
77:    }
78:
79:
80:# ---------------------------------------------------------------------------
81:# Per-bot readers
82:# ---------------------------------------------------------------------------
83:
84:
85:def _golddigger(dd: Path, now: datetime) -> dict[str, Any]:
86:    state = load_json(dd / "golddigger_state.json", default=None)
87:    if state is None:
88:        return _unknown("golddigger", "GoldDigger")
89:    halted = bool(state.get("halted"))
90:    reason = (state.get("halted_reason") or "").strip()
91:    position = state.get("position")
92:    if halted:
93:        return _emit(
94:            "golddigger", "GoldDigger", "HALTED",
95:            reason or "halted (no reason recorded)",
96:            position=position, state=state,
97:        )
98:    if position:
99:        return _emit(
100:            "golddigger", "GoldDigger", "TRADING",
101:            "position open",
102:            position=position, state=state,
103:        )
104:    if not _in_session(now):
105:        return _emit(
106:            "golddigger", "GoldDigger", "OUTSIDE_HOURS",
107:            _next_open_hint(now),
108:            position=None, state=state,
109:        )
110:    return _emit(
111:        "golddigger", "GoldDigger", "SCANNING",
112:        "in session, no entry signal yet",
113:        position=None, state=state,
114:    )
115:
116:
117:def _elongir(dd: Path, now: datetime) -> dict[str, Any]:
118:    state = load_json(dd / "elongir_state.json", default=None)
119:    if state is None:
120:        return _unknown("elongir", "Elongir")
121:    halted = bool(state.get("halted"))
122:    reason = (state.get("halted_reason") or "").strip()
123:    position = state.get("position")
124:    if halted:
125:        return _emit(
126:            "elongir", "Elongir", "HALTED",
127:            reason or "halted (no reason recorded)",
128:            position=position, state=state,
129:        )
130:    if position:
131:        return _emit(
132:            "elongir", "Elongir", "TRADING",
133:            "position open",
134:            position=position, state=state,
135:        )
136:    if not _in_session(now):
137:        return _emit(
138:            "elongir", "Elongir", "OUTSIDE_HOURS",
139:            _next_open_hint(now),
140:            position=None, state=state,
141:        )
142:    return _emit(
143:        "elongir", "Elongir", "SCANNING",
144:        "in session, no dip detected",
145:        position=None, state=state,
146:    )
147:
148:
149:def _metals(dd: Path, now: datetime) -> dict[str, Any]:
150:    state = load_json(dd / "metals_swing_state.json", default=None)
151:    guard = load_json(dd / "metals_guard_state.json", default={}) or {}
152:    if state is None:
153:        return _unknown("metals", "Metals swing")
154:    positions = state.get("positions") or []
155:    has_position = bool(positions) if isinstance(positions, list) else bool(positions)
156:    if has_position:
157:        return _emit(
158:            "metals", "Metals swing", "TRADING",
159:            f"holding {len(positions)} position(s)" if isinstance(positions, list) else "position open",
160:            position=positions, state=state,
161:        )
162:    if not _in_session(now):
163:        return _emit(
164:            "metals", "Metals swing", "OUTSIDE_HOURS",
165:            _next_open_hint(now),
166:            position=None, state=state,
167:        )
168:    consecutive_losses = guard.get("consecutive_losses") or state.get("consecutive_losses") or 0
169:    last_buy_ts = state.get("last_buy_ts")
170:    reason = "in session, no signal"
171:    if consecutive_losses and consecutive_losses >= 3:
172:        reason = f"in session, {consecutive_losses} consecutive losses (caution)"
173:    elif last_buy_ts:
174:        reason = "in session, between trades"
175:    return _emit(
176:        "metals", "Metals swing", "SCANNING",
177:        reason,
178:        position=None, state=state,
179:    )
180:
181:
182:def _fishing(dd: Path, now: datetime) -> dict[str, Any]:
183:    state = load_json(dd / "fish_engine_state.json", default=None)
184:    if state is None:
185:        return _unknown("fishing", "Fishing engine")
186:    position = state.get("position")
187:    if position:
188:        return _emit(
189:            "fishing", "Fishing engine", "TRADING",
190:            "position open",
191:            position=position, state=state,
192:        )
193:    last_trade_ts = state.get("last_trade_ts")
194:    cooldown_s = state.get("cooldown_seconds") or 0
195:    if last_trade_ts and cooldown_s:
196:        try:
197:            last = datetime.fromtimestamp(float(last_trade_ts), tz=UTC)
198:            cool_until = last.timestamp() + float(cooldown_s)
199:            remaining = cool_until - now.timestamp()
200:            if remaining > 0:
201:                losses = state.get("consecutive_losses") or 0
202:                detail = f"{int(remaining)}s remaining"
203:                if losses:
204:                    detail = f"{losses} losses, {detail}"
205:                return _emit(
206:                    "fishing", "Fishing engine", "COOLDOWN",
207:                    detail,
208:                    position=None, state=state,
209:                )
210:        except Exception:
211:            pass
212:    if not _in_session(now):
213:        return _emit(
214:            "fishing", "Fishing engine", "OUTSIDE_HOURS",
215:            _next_open_hint(now),
216:            position=None, state=state,
217:        )
218:    return _emit(
219:        "fishing", "Fishing engine", "SCANNING",
220:        f"mode={state.get('mode') or 'idle'}",
221:        position=None, state=state,
222:    )
223:
224:
225:# ---------------------------------------------------------------------------
226:# Helpers
227:# ---------------------------------------------------------------------------
228:
229:
230:def _emit(
231:    bot: str,
232:    label: str,
233:    state_name: str,
234:    reason: str,
235:    *,
236:    position: Any,
237:    state: dict,
238:) -> dict[str, Any]:
239:    return {
240:        "bot": bot,
241:        "label": label,
242:        "state": state_name,
243:        "reason": reason,
244:        "position": position,
245:        "stats": _extract_stats(state),
246:    }
247:
248:
249:def _unknown(bot: str, label: str) -> dict[str, Any]:
250:    return {
251:        "bot": bot,
252:        "label": label,
253:        "state": "UNKNOWN",
254:        "reason": "state file missing or unreadable",
255:        "position": None,
256:        "stats": {},
257:        "error": "state file not found",
258:    }
259:
260:
261:def _extract_stats(state: dict) -> dict[str, Any]:
262:    out: dict[str, Any] = {}
263:    for k in (
264:        "daily_pnl",
265:        "daily_trades",
266:        "total_trades",
267:        "total_pnl",
268:        "consecutive_losses",
269:        "cash_sek",
270:        "equity_sek",
271:    ):
272:        if k in state:
273:            out[k] = state[k]
274:    return out
275:
276:
277:def _in_session(now_utc: datetime) -> bool:
278:    """True iff Europe/Stockholm wall-clock time is within the warrant
279:    session (Mon–Fri, 15:30–21:55 inclusive of open, exclusive of close).
280:
281:    Codex P1 finding 2026-05-04: weekday check matters — Saturday and
282:    Sunday at 16:00 local time would otherwise read as session_open and
283:    bots would render as SCANNING when they're correctly idle.
284:    """
285:    local = now_utc.astimezone(SESSION_TZ)
286:    if local.weekday() >= 5:  # Saturday=5, Sunday=6
287:        return False
288:    t = local.timetz().replace(tzinfo=None)
289:    return SESSION_OPEN <= t < SESSION_CLOSE
290:
291:
292:def _next_open_hint(now_utc: datetime) -> str:
293:    """Human-readable 'next 08:30 CEST in 2h 14m'.
294:
295:    Rolls forward to the next weekday open and uses the *target* date's
296:    tzname() so the suffix flips between CET and CEST automatically
297:    across the DST boundary. The open-time string is read from
298:    SESSION_OPEN — bug-fix 2026-05-11: previously hard-coded "15:30"
299:    which lied to users once the session moved to 08:30.
300:    """
301:    from datetime import timedelta
302:
303:    local_now = now_utc.astimezone(SESSION_TZ)
304:    target = local_now.replace(
305:        hour=SESSION_OPEN.hour, minute=SESSION_OPEN.minute,
306:        second=0, microsecond=0,
307:    )
308:    if local_now >= target:
309:        target = target + timedelta(days=1)
310:    while target.weekday() >= 5:  # skip weekend(s) into next Monday
311:        target = target + timedelta(days=1)
312:    delta = target - local_now
313:    hours = int(delta.total_seconds() // 3600)
314:    mins = int((delta.total_seconds() % 3600) // 60)
315:    zone = target.tzname() or "Stockholm"
316:    return (
317:        f"next {SESSION_OPEN.hour:02d}:{SESSION_OPEN.minute:02d} "
318:        f"{zone} in {hours}h {mins:02d}m"
319:    )
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 775ms:
dashboard/house_blueprint.py:54:def _house_root() -> Path:
dashboard/house_blueprint.py:62:def _runs_dir() -> Path:
dashboard/house_blueprint.py:66:def _heatmap_path() -> Path:
dashboard/house_blueprint.py:77:def _validate_run_id(run_id: str) -> str:
dashboard/house_blueprint.py:83:def _validate_slug(slug: str) -> str:
dashboard/house_blueprint.py:96:def _list_runs() -> list[dict]:
dashboard/house_blueprint.py:124:def _resolve_md(run_id: str, basename: str) -> Optional[Path]:
dashboard/house_blueprint.py:192:def _shell(title: str, body_html: str, breadcrumbs: list[tuple[str, str]]) -> str:
dashboard/house_blueprint.py:211:def _render_markdown(text: str) -> str:
dashboard/house_blueprint.py:224:@bp.route("/")
dashboard/house_blueprint.py:226:def index():
dashboard/house_blueprint.py:243:@bp.route("/runs")
dashboard/house_blueprint.py:245:def runs_list():
dashboard/house_blueprint.py:273:@bp.route("/runs/<run_id>")
dashboard/house_blueprint.py:275:def run_detail(run_id: str):
dashboard/house_blueprint.py:312:@bp.route("/runs/<run_id>/_manifest.json")
dashboard/house_blueprint.py:314:def run_manifest(run_id: str):
dashboard/house_blueprint.py:322:@bp.route("/runs/<run_id>/<slug>")
dashboard/house_blueprint.py:324:def candidate_detail(run_id: str, slug: str):
dashboard/house_blueprint.py:348:@bp.route("/runs/<run_id>/<slug>/raw")
dashboard/house_blueprint.py:350:def candidate_raw(run_id: str, slug: str):
dashboard/house_blueprint.py:359:@bp.route("/heatmap")
dashboard/house_blueprint.py:361:def heatmap():
dashboard/house_blueprint.py:373:@bp.route("/api/runs")
dashboard/house_blueprint.py:375:def api_runs():
dashboard/house_blueprint.py:379:@bp.route("/api/runs/<run_id>")
dashboard/house_blueprint.py:381:def api_run(run_id: str):
dashboard/house_blueprint.py:393:@bp.route("/api/runs/<run_id>/<slug>")
dashboard/house_blueprint.py:395:def api_candidate(run_id: str, slug: str):
dashboard/app.py:20:def _json_safe(value):
dashboard/app.py:36:    def dumps(self, obj, **kwargs):
dashboard/app.py:53:def add_cors_headers(response):
dashboard/app.py:84:def _cached_read(key, ttl, read_fn):
dashboard/app.py:101:def _read_json(path, ttl=_DEFAULT_TTL):
dashboard/app.py:105:def _read_jsonl(path, limit=100, ttl=_DEFAULT_TTL):
dashboard/app.py:133:def _read_tail_with_growth(path, limit):
dashboard/app.py:171:def _get_config():
dashboard/app.py:175:def _parse_limit_arg(name, default, max_value):
dashboard/app.py:184:def _iter_latest_dict_entries(path, read_limit):
dashboard/app.py:192:def _parse_iso8601(value):
dashboard/app.py:205:def _stockholm_now():
dashboard/app.py:209:def _hours_until_stockholm_close(now=None, close_hour=21, close_minute=30):
dashboard/app.py:223:def _is_number(value):
dashboard/app.py:227:def _round_or_none(value, digits=2):
dashboard/app.py:231:def _normalize_golddigger_position(raw_position, latest_log):
dashboard/app.py:268:def _normalize_golddigger_log_entry(entry):
dashboard/app.py:279:def _normalize_golddigger_trade_entry(entry):
dashboard/app.py:296:def _normalize_golddigger_state(state, log_entries):
dashboard/app.py:347:def _normalize_metals_llm_predictions(raw_llm):
dashboard/app.py:398:def _normalize_metals_forecast_signals(raw_llm):
dashboard/app.py:420:def _normalize_metals_decisions(decisions):
dashboard/app.py:441:def _drawdown_level_from_pct(drawdown_pct):
dashboard/app.py:451:def _normalize_metals_risk(risk):
dashboard/app.py:471:def _normalize_metals_context(context):
dashboard/app.py:479:def _merge_missing_structure(primary, fallback):
dashboard/app.py:502:def _build_metals_context_fallback(decisions):
dashboard/app.py:666:def _aggregate_accuracy_bucket(bucket):
dashboard/app.py:686:def _build_local_llm_trend_point(entry, ticker=None):
dashboard/app.py:755:@app.route("/")
dashboard/app.py:757:def index():
dashboard/app.py:767:@app.route("/legacy")
dashboard/app.py:769:def index_legacy():
dashboard/app.py:777:@app.route("/logout")
dashboard/app.py:778:def logout():
dashboard/app.py:809:@app.route("/api/summary")
dashboard/app.py:811:def api_summary():
dashboard/app.py:825:@app.route("/api/signals")
dashboard/app.py:827:def api_signals():
dashboard/app.py:834:@app.route("/api/portfolio")
dashboard/app.py:836:def api_portfolio():
dashboard/app.py:843:@app.route("/api/portfolio-bold")
dashboard/app.py:845:def api_portfolio_bold():
dashboard/app.py:852:@app.route("/api/grid-fisher")
dashboard/app.py:854:def api_grid_fisher():
dashboard/app.py:875:@app.route("/api/mstr_loop")
dashboard/app.py:877:def api_mstr_loop():
dashboard/app.py:922:@app.route("/api/invocations")
dashboard/app.py:924:def api_invocations():
dashboard/app.py:929:@app.route("/api/telegrams")
dashboard/app.py:931:def api_telegrams():
dashboard/app.py:956:@app.route("/api/signal-log")
dashboard/app.py:958:def api_signal_log():
dashboard/app.py:967:@app.route("/api/accuracy")
dashboard/app.py:969:def api_accuracy():
dashboard/app.py:992:        def _enrich_signals(signals_dict):
dashboard/app.py:1047:@app.route("/api/iskbets")
dashboard/app.py:1049:def api_iskbets():
dashboard/app.py:1055:@app.route("/api/lora-status")
dashboard/app.py:1057:def api_lora_status():
dashboard/app.py:1067:@app.route("/api/validate-portfolio", methods=["POST"])
dashboard/app.py:1069:def api_validate_portfolio():
dashboard/app.py:1095:@app.route("/api/equity-curve")
dashboard/app.py:1097:def api_equity_curve():
dashboard/app.py:1110:@app.route("/api/signal-heatmap")
dashboard/app.py:1112:def api_signal_heatmap():
dashboard/app.py:1209:@app.route("/api/triggers")
dashboard/app.py:1211:def api_triggers():
dashboard/app.py:1217:@app.route("/api/accuracy-history")
dashboard/app.py:1219:def api_accuracy_history():
dashboard/app.py:1243:@app.route("/api/local-llm-trends")
dashboard/app.py:1245:def api_local_llm_trends():
dashboard/app.py:1268:@app.route("/api/metals-accuracy")
dashboard/app.py:1270:def api_metals_accuracy():
dashboard/app.py:1278:@app.route("/api/trades")
dashboard/app.py:1280:def api_trades():
dashboard/app.py:1309:@app.route("/api/decisions")
dashboard/app.py:1311:def api_decisions():
dashboard/app.py:1352:@app.route("/api/health")
dashboard/app.py:1354:def api_health():
dashboard/app.py:1368:@app.route("/api/warrants")
dashboard/app.py:1370:def api_warrants():
dashboard/app.py:1385:@app.route("/api/risk")
dashboard/app.py:1387:def api_risk():
dashboard/app.py:1405:@app.route("/api/metals")
dashboard/app.py:1407:def api_metals():
dashboard/app.py:1439:def _crypto_per_instrument(state: dict, ticker: str) -> dict:
dashboard/app.py:1451:def _crypto_decisions_for(decisions: list, ticker: str) -> list:
dashboard/app.py:1462:@app.route("/api/crypto")
dashboard/app.py:1464:def api_crypto():
dashboard/app.py:1493:@app.route("/api/btc")
dashboard/app.py:1495:def api_btc():
dashboard/app.py:1513:@app.route("/api/eth")
dashboard/app.py:1515:def api_eth():
dashboard/app.py:1533:@app.route("/api/loop_health")
dashboard/app.py:1535:def api_loop_health():
dashboard/app.py:1551:@app.route("/api/oil")
dashboard/app.py:1553:def api_oil():
dashboard/app.py:1589:@app.route("/api/mstr")
dashboard/app.py:1591:def api_mstr():
dashboard/app.py:1615:@app.route("/api/golddigger")
dashboard/app.py:1617:def api_golddigger():
dashboard/app.py:1639:@app.route("/api/market-health")
dashboard/app.py:1641:def api_market_health():
dashboard/app.py:1719:def _avanza_worker_loop() -> None:
dashboard/app.py:1741:def _ensure_avanza_worker() -> None:
dashboard/app.py:1755:def _avanza_account_snapshot() -> dict:
dashboard/app.py:1781:def _avanza_snapshot_impl() -> dict:
dashboard/app.py:1854:def _norm_order(raw: dict) -> dict:
dashboard/app.py:1868:def _stop_account(raw: dict) -> str:
dashboard/app.py:1875:def _norm_stop(raw: dict) -> dict:
dashboard/app.py:1892:@app.route("/api/avanza_account")
dashboard/app.py:1894:def api_avanza_account():
dashboard/app.py:1929:@app.route("/api/tradeable_assets")
dashboard/app.py:1931:def api_tradeable_assets():
dashboard/app.py:1981:@app.route("/api/system_status")
dashboard/app.py:1983:def api_system_status():
dashboard/app.py:2015:@app.route("/api/trading_status")
dashboard/app.py:2017:def api_trading_status():
dashboard/app.py:2058:def _serve_dual_stack(port: int = 5055) -> None:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 759ms:
dashboard/house_blueprint.py:108:            slugs = json.loads(manifest.read_text())
dashboard/house_blueprint.py:230:        # User auth-bootstrapped via ?token= — strip it from the URL.
dashboard/house_blueprint.py:231:        return redirect("/house", code=302)
dashboard/house_blueprint.py:240:    return redirect(f"/house/runs/{runs[0]['run_id']}", code=302)
dashboard/house_blueprint.py:280:    text = summary.read_text(encoding="utf-8")
dashboard/house_blueprint.py:288:        slugs = json.loads(manifest.read_text())
dashboard/house_blueprint.py:319:    return send_file(manifest, mimetype="application/json")
dashboard/house_blueprint.py:330:    text = md_path.read_text(encoding="utf-8")
dashboard/house_blueprint.py:356:    return send_file(raw, mimetype="application/json")
dashboard/house_blueprint.py:365:    return send_file(path, mimetype="text/html")
dashboard/house_blueprint.py:387:        slugs = json.loads(manifest.read_text())
dashboard/house_blueprint.py:401:    return send_file(raw, mimetype="application/json")
dashboard/export_static.py:64:        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
dashboard/export_static.py:81:    out_dir.mkdir(parents=True, exist_ok=True)
dashboard/export_static.py:93:                    req_route = f"{req_route}{sep}token={token}"
dashboard/export_static.py:106:                with open(dest, "w", encoding="utf-8") as f:
dashboard/auth.py:62:        with open(CONFIG_PATH, encoding="utf-8") as fp:
dashboard/auth.py:63:            data = json.load(fp)
dashboard/auth.py:92:    response.set_cookie(
dashboard/auth.py:107:      0. Cf-Access-Authenticated-User-Email header — Cloudflare Access has
dashboard/auth.py:110:      2. ?token= query param — for first-visit-from-a-new-browser.
dashboard/auth.py:111:      3. Authorization: Bearer header — for CLI / script clients.
dashboard/auth.py:126:        # Cf-Access-* headers at its edge and re-injects them only after
dashboard/auth.py:132:        cf_email = request.headers.get("Cf-Access-Authenticated-User-Email")
dashboard/auth.py:133:        cf_jwt = request.headers.get("Cf-Access-Jwt-Assertion")
dashboard/auth.py:147:        # 3. Authorization: Bearer (CLI / script clients — no cookie set
dashboard/auth.py:149:        auth_header = request.headers.get("Authorization", "")
dashboard/auth.py:158:                "Visit /?token=YOUR_TOKEN once to set a 1-year rolling auth "
dashboard/app.py:58:    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
dashboard/app.py:59:    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
dashboard/app.py:758:    # If the user arrived via ?token=XXX, the cookie was just set in
dashboard/app.py:763:        return redirect("/", code=302)
dashboard/app.py:773:        return redirect("/legacy", code=302)
dashboard/app.py:791:    response = redirect("/", code=302)
dashboard/app.py:793:    response.set_cookie(
dashboard/app.py:898:            with open(poll_path, encoding="utf-8") as f:
dashboard/app.py:902:                            out["last_poll"] = _json.loads(line)
dashboard/app.py:910:            with open(trades_path, encoding="utf-8") as f:
dashboard/app.py:914:                            out["last_trade"] = _json.loads(line)
dashboard/app.py:1067:@app.route("/api/validate-portfolio", methods=["POST"])
dashboard/app.py:1903:    `?force=1` bypasses the TTL cache so the user's manual Refresh
dashboard/app.py:1977:# cache mirrors the _AVANZA_CACHE pattern; ?force=1 bypasses for the
dashboard/app.py:1993:    fresher payload, and ``?force=1`` won't lose its refresh behind
dashboard/app.py:2051:# to app.py. The sys.modules alias hack added 2026-05-02 has been removed.
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

2026-05-11T15:29:20.949498Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
portfolio/backtester.py:129:            # Filter out disabled signals for old consensus too
portfolio/vector_memory.py:4:no sentence-transformers needed). Lazy-init singleton. Embeds journal
portfolio/vector_memory.py:15:Defaults to disabled — graceful fallback if chromadb is not installed.
portfolio/vector_memory.py:272:    with open(JOURNAL_FILE, encoding="utf-8") as f:
portfolio/vector_memory.py:278:                entries.append(json.loads(line))
portfolio/vector_memory.py:285:    """Reset the singleton (for testing)."""
scripts/check_critical_errors.py:4:the STARTUP CHECK block in CLAUDE.md). Prints a compact summary of
scripts/check_critical_errors.py:42:    for line in journal.read_text(encoding="utf-8", errors="replace").splitlines():
scripts/check_critical_errors.py:47:            entries.append(json.loads(line))
portfolio/file_utils.py:10:# Cross-platform file-locking primitives for `atomic_append_jsonl`.
portfolio/file_utils.py:11:# Same pattern as `portfolio/process_lock.py`.
portfolio/file_utils.py:24:def atomic_write_text(path, text, encoding="utf-8"):
portfolio/file_utils.py:27:    Same safety guarantees as atomic_write_json: fsync before replace,
portfolio/file_utils.py:34:        with os.fdopen(fd, "w", encoding=encoding) as f:
portfolio/file_utils.py:45:def atomic_write_json(path, data, indent=2, ensure_ascii=True):
portfolio/file_utils.py:55:        with os.fdopen(fd, "w", encoding="utf-8") as f:
portfolio/file_utils.py:66:def load_json(path, default=None):
portfolio/file_utils.py:70:    Handles OSError (permission denied, locked files) gracefully on Windows.
portfolio/file_utils.py:75:        return json.loads(path.read_text(encoding="utf-8"))
portfolio/file_utils.py:79:        # BUG-139: PermissionError (file locked by antivirus/another process)
portfolio/file_utils.py:81:        logger.debug("load_json: OS error reading %s, returning default", path.name)
portfolio/file_utils.py:85:        logger.warning("load_json: corrupt JSON in %s, returning default", path.name)
portfolio/file_utils.py:92:    Unlike load_json(), this function does NOT silently return defaults.
portfolio/file_utils.py:101:    return json.loads(path.read_text(encoding="utf-8"))
portfolio/file_utils.py:104:def load_jsonl(path, limit=None):
portfolio/file_utils.py:117:        f = open(path, encoding="utf-8")
portfolio/file_utils.py:121:        logger.warning("load_jsonl: cannot open %s: %s", path.name, e)
portfolio/file_utils.py:129:                container.append(json.loads(line))
portfolio/file_utils.py:136:def load_jsonl_tail(path, max_entries=500, tail_bytes=512_000):
portfolio/file_utils.py:139:    Much more efficient than load_jsonl(limit=N) for large files because
portfolio/file_utils.py:161:        with open(path, "rb") as f:
portfolio/file_utils.py:189:                entries.append(json.loads(line))
portfolio/file_utils.py:193:        logger.debug("load_jsonl_tail failed for %s: %s", path.name, e)
portfolio/file_utils.py:203:def jsonl_sidecar_lock(path):
portfolio/file_utils.py:204:    """Yield while holding an exclusive sidecar lock keyed off *path*.
portfolio/file_utils.py:206:    Same locking primitive that ``atomic_append_jsonl`` uses, exposed as
portfolio/file_utils.py:209:    in-flight appends. Lock file is ``<path.parent>/.<path.name>.lock``;
portfolio/file_utils.py:210:    a single-byte range is locked exclusively.
portfolio/file_utils.py:214:    * **Sidecar (not target):** locking the target file itself is racy
portfolio/file_utils.py:216:      the empty-file ``msvcrt.locking(fd, LK_LOCK, 1)`` failure path
portfolio/file_utils.py:217:      and interleave. A pre-seeded sidecar guarantees a lockable byte
portfolio/file_utils.py:219:    * **Windows + POSIX:** ``msvcrt.locking`` blocks on contention on
portfolio/file_utils.py:220:      Windows; ``fcntl.flock`` blocks on POSIX. Both release on close.
portfolio/file_utils.py:223:    ``with`` block — read, write, fsync, rename. Appends that arrive
portfolio/file_utils.py:231:    lock_path = path.parent / f".{path.name}.lock"
portfolio/file_utils.py:232:    if not lock_path.exists():
portfolio/file_utils.py:234:            with open(lock_path, "ab") as lf:
portfolio/file_utils.py:238:            pass  # best-effort; lock open below will retry
portfolio/file_utils.py:240:    with open(lock_path, "rb+") as lock_f:
portfolio/file_utils.py:241:        lfd = lock_f.fileno()
portfolio/file_utils.py:242:        win_locked = False
portfolio/file_utils.py:246:                _msvcrt.locking(lfd, _msvcrt.LK_LOCK, 1)  # blocking
portfolio/file_utils.py:247:                win_locked = True
portfolio/file_utils.py:249:                _fcntl.flock(lfd, _fcntl.LOCK_EX)
portfolio/file_utils.py:252:            if win_locked and _msvcrt is not None:
portfolio/file_utils.py:255:                    _msvcrt.locking(lfd, _msvcrt.LK_UNLCK, 1)
portfolio/file_utils.py:258:            # fcntl.flock releases automatically on close.
portfolio/file_utils.py:261:def atomic_append_jsonl(path, entry):
portfolio/file_utils.py:265:    Now built on :func:`jsonl_sidecar_lock` so the lock contract is
portfolio/file_utils.py:280:    with jsonl_sidecar_lock(path):
portfolio/file_utils.py:281:        with open(path, "ab") as f:
portfolio/file_utils.py:287:def atomic_write_jsonl(path, entries):
portfolio/file_utils.py:296:        with os.fdopen(fd, "w", encoding="utf-8") as f:
portfolio/file_utils.py:330:        with open(path, "rb") as f:
portfolio/file_utils.py:340:            entry = json.loads(line)
portfolio/file_utils.py:361:        f = open(path, encoding="utf-8")
portfolio/file_utils.py:371:                json.loads(stripped)
portfolio/file_utils.py:382:        with os.fdopen(fd, "w", encoding="utf-8") as f:
portfolio/shadow_registry.py:49:from portfolio.file_utils import atomic_write_json, load_json
portfolio/shadow_registry.py:68:    data = load_json(str(p), default=None)
portfolio/shadow_registry.py:78:    atomic_write_json(str(p), data)
portfolio/config_validator.py:9:from portfolio.file_utils import load_json
portfolio/config_validator.py:20:# ``binance.key`` / ``binance.secret`` was blocking main-loop startup on
portfolio/config_validator.py:66:    config = load_json(CONFIG_FILE)
scripts/fix_agent_dispatcher.py:8:unresolved entries, respects cooldown and kill-switch, and invokes
scripts/fix_agent_dispatcher.py:41:KILL_SWITCH = DATA_DIR / "fix_agent.disabled"
scripts/fix_agent_dispatcher.py:46:# Exponential backoff on consecutive failures.
scripts/fix_agent_dispatcher.py:47:BACKOFF_SCHEDULE_S = [1800, 7200, 43200]  # 30m → 2h → 12h, then disabled
scripts/fix_agent_dispatcher.py:58:AGENT_ALLOWED_TOOLS = "Read,Edit,Bash"
scripts/fix_agent_dispatcher.py:84:    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
scripts/fix_agent_dispatcher.py:89:            entries.append(json.loads(line))
scripts/fix_agent_dispatcher.py:101:    default: a 1-hour blanket cooldown across all categories. Without
scripts/fix_agent_dispatcher.py:102:    that guard, a corrupt state file would clear all cooldowns and let
scripts/fix_agent_dispatcher.py:111:        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
scripts/fix_agent_dispatcher.py:126:                    "blanket cooldown to prevent runaway spawns. Inspect "
scripts/fix_agent_dispatcher.py:133:        # Conservative default: block ALL spawns for 1 hour via the
scripts/fix_agent_dispatcher.py:134:        # global blocked_until_global field. check_gates honours this
scripts/fix_agent_dispatcher.py:139:            "blocked_until_global": (_now() + timedelta(hours=1)).isoformat(),
scripts/fix_agent_dispatcher.py:149:    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
scripts/fix_agent_dispatcher.py:184:    helper the main loop uses (``portfolio.file_utils.atomic_append_jsonl``).
scripts/fix_agent_dispatcher.py:191:    falling back to a non-atomic ``open("a")`` — silent corruption of the
scripts/fix_agent_dispatcher.py:197:        from portfolio.file_utils import atomic_append_jsonl
scripts/fix_agent_dispatcher.py:205:    atomic_append_jsonl(CRITICAL_ERRORS_LOG, entry)
scripts/fix_agent_dispatcher.py:209:# Gating — kill switch, cooldown, recursion
scripts/fix_agent_dispatcher.py:215:    reason: str  # machine-readable short tag, e.g. "cooldown" / "ok"
scripts/fix_agent_dispatcher.py:228:    var taken at run() startup. Passing it explicitly (rather than
scripts/fix_agent_dispatcher.py:236:        return GateDecision(False, "disabled_by_kill_switch")
scripts/fix_agent_dispatcher.py:238:    # Global block — set by _load_state when the state file was corrupt.
scripts/fix_agent_dispatcher.py:239:    global_block = _parse_iso(state.get("blocked_until_global"))
scripts/fix_agent_dispatcher.py:240:    if global_block and global_block > now:
scripts/fix_agent_dispatcher.py:241:        return GateDecision(False, "global_cooldown")
scripts/fix_agent_dispatcher.py:249:    blocked_until = _parse_iso(cat_state.get("blocked_until"))
scripts/fix_agent_dispatcher.py:250:    if blocked_until and blocked_until > now:
scripts/fix_agent_dispatcher.py:251:        return GateDecision(False, "cooldown")
scripts/fix_agent_dispatcher.py:259:    """Bump cooldown + consecutive_failures for the category."""
scripts/fix_agent_dispatcher.py:269:        entry["blocked_until"] = (now + timedelta(seconds=SELF_HEAL_COOLDOWN_S)).isoformat()
scripts/fix_agent_dispatcher.py:276:            # Beyond the schedule: effectively disabled for 10 years. User
scripts/fix_agent_dispatcher.py:279:            entry["blocked_until"] = (now + timedelta(days=3650)).isoformat()
scripts/fix_agent_dispatcher.py:281:            entry["blocked_until"] = (now + timedelta(seconds=BACKOFF_SCHEDULE_S[idx])).isoformat()
scripts/fix_agent_dispatcher.py:333:def run(
scripts/fix_agent_dispatcher.py:402:        # call so the child Claude subprocess inherits depth+1 (blocking any
scripts/fix_agent_dispatcher.py:458:        return run(dry_run=args.dry_run, lookback_h=args.lookback_h)
portfolio/subprocess_utils.py:4:- run_safe(): Drop-in subprocess.run() replacement that uses Windows Job Objects
portfolio/subprocess_utils.py:24:    Drop-in replacement for subprocess.run().  On Windows, creates a Job Object
portfolio/subprocess_utils.py:28:    Falls back to plain subprocess.run() on non-Windows or if Job Object
portfolio/subprocess_utils.py:32:    others accepted by subprocess.Popen / subprocess.run).
portfolio/subprocess_utils.py:35:        return subprocess.run(cmd, **kwargs)
portfolio/subprocess_utils.py:40:        logger.debug("Job Object creation failed (%s), falling back to subprocess.run", exc)
portfolio/subprocess_utils.py:41:        return subprocess.run(cmd, **kwargs)
portfolio/subprocess_utils.py:54:    job = kernel32.CreateJobObjectW(None, None)
portfolio/subprocess_utils.py:56:        raise OSError("CreateJobObjectW failed")
portfolio/subprocess_utils.py:122:            popen_kwargs["stdout"] = subprocess.PIPE
portfolio/subprocess_utils.py:123:            popen_kwargs["stderr"] = subprocess.PIPE
portfolio/subprocess_utils.py:127:            popen_kwargs["stdin"] = subprocess.PIPE
portfolio/subprocess_utils.py:129:        proc = subprocess.Popen(cmd, **popen_kwargs)
portfolio/subprocess_utils.py:132:            kernel32.AssignProcessToJobObject(job, int(proc._handle))
portfolio/subprocess_utils.py:141:        except subprocess.TimeoutExpired:
portfolio/subprocess_utils.py:146:        return subprocess.CompletedProcess(
portfolio/subprocess_utils.py:159:    Like subprocess.Popen(), but assigns the child to a Job Object with
portfolio/subprocess_utils.py:168:    proc = subprocess.Popen(cmd, **kwargs)
portfolio/subprocess_utils.py:175:        kernel32.AssignProcessToJobObject(job, int(proc._handle))
portfolio/subprocess_utils.py:215:        result = subprocess.run(
portfolio/subprocess_utils.py:216:            ["wmic", "process", "where",
portfolio/subprocess_utils.py:239:            subprocess.run(
portfolio/subprocess_utils.py:240:                ["taskkill", "/F", "/PID", str(pid)],
portfolio/subprocess_utils.py:269:        'powershell.exe -NoProfile -Command "'
portfolio/subprocess_utils.py:275:        result = subprocess.run(
portfolio/subprocess_utils.py:280:            shell=True,
portfolio/subprocess_utils.py:290:        data = json.loads(result.stdout.strip())
portfolio/process_lock.py:1:"""Cross-platform helpers for non-blocking singleton process locks."""
portfolio/process_lock.py:22:def acquire_lock_file(
portfolio/process_lock.py:23:    lock_path: str | Path,
portfolio/process_lock.py:28:    """Acquire a non-blocking file lock and return the open handle.
portfolio/process_lock.py:30:    Returns None if another process already holds the lock.
portfolio/process_lock.py:32:    path = Path(lock_path)
portfolio/process_lock.py:36:    fh = path.open("a+", encoding="utf-8")
portfolio/process_lock.py:38:        _lock_file(fh)
portfolio/process_lock.py:43:    _write_lock_metadata(fh, owner=owner, metadata=metadata)
portfolio/process_lock.py:47:def release_lock_file(fh: IO[str] | None) -> None:
portfolio/process_lock.py:48:    """Release a previously acquired lock handle."""
portfolio/process_lock.py:52:        _unlock_file(fh)
portfolio/process_lock.py:60:def _lock_file(fh: IO[str]) -> None:
portfolio/process_lock.py:63:        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
portfolio/process_lock.py:66:        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
portfolio/process_lock.py:69:        "No file locking mechanism available (neither msvcrt nor fcntl). "
portfolio/process_lock.py:74:def _unlock_file(fh: IO[str]) -> None:
portfolio/process_lock.py:77:        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
portfolio/process_lock.py:80:        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
portfolio/process_lock.py:83:def _write_lock_metadata(
portfolio/message_store.py:27:from portfolio.file_utils import atomic_append_jsonl
portfolio/message_store.py:102:    atomic_append_jsonl(MESSAGES_FILE, entry)
portfolio/message_store.py:193:    # Per-category blocklist: mute specific categories
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: scripts/win/*.bat: The filename, directory name, or volume label syntax is incorrect. (os error 123)
rg: scripts/win/*.ps1: The filename, directory name, or volume label syntax is incorrect. (os error 123)
rg: conftest.py: The system cannot find the file specified. (os error 2)

 exited 1 in 786ms:
portfolio/backtester.py:129:            # Filter out disabled signals for old consensus too
portfolio/vector_memory.py:4:no sentence-transformers needed). Lazy-init singleton. Embeds journal
portfolio/vector_memory.py:15:Defaults to disabled — graceful fallback if chromadb is not installed.
portfolio/vector_memory.py:272:    with open(JOURNAL_FILE, encoding="utf-8") as f:
portfolio/vector_memory.py:278:                entries.append(json.loads(line))
portfolio/vector_memory.py:285:    """Reset the singleton (for testing)."""
scripts/check_critical_errors.py:4:the STARTUP CHECK block in CLAUDE.md). Prints a compact summary of
scripts/check_critical_errors.py:42:    for line in journal.read_text(encoding="utf-8", errors="replace").splitlines():
scripts/check_critical_errors.py:47:            entries.append(json.loads(line))
portfolio/file_utils.py:10:# Cross-platform file-locking primitives for `atomic_append_jsonl`.
portfolio/file_utils.py:11:# Same pattern as `portfolio/process_lock.py`.
portfolio/file_utils.py:24:def atomic_write_text(path, text, encoding="utf-8"):
portfolio/file_utils.py:27:    Same safety guarantees as atomic_write_json: fsync before replace,
portfolio/file_utils.py:34:        with os.fdopen(fd, "w", encoding=encoding) as f:
portfolio/file_utils.py:45:def atomic_write_json(path, data, indent=2, ensure_ascii=True):
portfolio/file_utils.py:55:        with os.fdopen(fd, "w", encoding="utf-8") as f:
portfolio/file_utils.py:66:def load_json(path, default=None):
portfolio/file_utils.py:70:    Handles OSError (permission denied, locked files) gracefully on Windows.
portfolio/file_utils.py:75:        return json.loads(path.read_text(encoding="utf-8"))
portfolio/file_utils.py:79:        # BUG-139: PermissionError (file locked by antivirus/another process)
portfolio/file_utils.py:81:        logger.debug("load_json: OS error reading %s, returning default", path.name)
portfolio/file_utils.py:85:        logger.warning("load_json: corrupt JSON in %s, returning default", path.name)
portfolio/file_utils.py:92:    Unlike load_json(), this function does NOT silently return defaults.
portfolio/file_utils.py:101:    return json.loads(path.read_text(encoding="utf-8"))
portfolio/file_utils.py:104:def load_jsonl(path, limit=None):
portfolio/file_utils.py:117:        f = open(path, encoding="utf-8")
portfolio/file_utils.py:121:        logger.warning("load_jsonl: cannot open %s: %s", path.name, e)
portfolio/file_utils.py:129:                container.append(json.loads(line))
portfolio/file_utils.py:136:def load_jsonl_tail(path, max_entries=500, tail_bytes=512_000):
portfolio/file_utils.py:139:    Much more efficient than load_jsonl(limit=N) for large files because
portfolio/file_utils.py:161:        with open(path, "rb") as f:
portfolio/file_utils.py:189:                entries.append(json.loads(line))
portfolio/file_utils.py:193:        logger.debug("load_jsonl_tail failed for %s: %s", path.name, e)
portfolio/file_utils.py:203:def jsonl_sidecar_lock(path):
portfolio/file_utils.py:204:    """Yield while holding an exclusive sidecar lock keyed off *path*.
portfolio/file_utils.py:206:    Same locking primitive that ``atomic_append_jsonl`` uses, exposed as
portfolio/file_utils.py:209:    in-flight appends. Lock file is ``<path.parent>/.<path.name>.lock``;
portfolio/file_utils.py:210:    a single-byte range is locked exclusively.
portfolio/file_utils.py:214:    * **Sidecar (not target):** locking the target file itself is racy
portfolio/file_utils.py:216:      the empty-file ``msvcrt.locking(fd, LK_LOCK, 1)`` failure path
portfolio/file_utils.py:217:      and interleave. A pre-seeded sidecar guarantees a lockable byte
portfolio/file_utils.py:219:    * **Windows + POSIX:** ``msvcrt.locking`` blocks on contention on
portfolio/file_utils.py:220:      Windows; ``fcntl.flock`` blocks on POSIX. Both release on close.
portfolio/file_utils.py:223:    ``with`` block — read, write, fsync, rename. Appends that arrive
portfolio/file_utils.py:231:    lock_path = path.parent / f".{path.name}.lock"
portfolio/file_utils.py:232:    if not lock_path.exists():
portfolio/file_utils.py:234:            with open(lock_path, "ab") as lf:
portfolio/file_utils.py:238:            pass  # best-effort; lock open below will retry
portfolio/file_utils.py:240:    with open(lock_path, "rb+") as lock_f:
portfolio/file_utils.py:241:        lfd = lock_f.fileno()
portfolio/file_utils.py:242:        win_locked = False
portfolio/file_utils.py:246:                _msvcrt.locking(lfd, _msvcrt.LK_LOCK, 1)  # blocking
portfolio/file_utils.py:247:                win_locked = True
portfolio/file_utils.py:249:                _fcntl.flock(lfd, _fcntl.LOCK_EX)
portfolio/file_utils.py:252:            if win_locked and _msvcrt is not None:
portfolio/file_utils.py:255:                    _msvcrt.locking(lfd, _msvcrt.LK_UNLCK, 1)
portfolio/file_utils.py:258:            # fcntl.flock releases automatically on close.
portfolio/file_utils.py:261:def atomic_append_jsonl(path, entry):
portfolio/file_utils.py:265:    Now built on :func:`jsonl_sidecar_lock` so the lock contract is
portfolio/file_utils.py:280:    with jsonl_sidecar_lock(path):
portfolio/file_utils.py:281:        with open(path, "ab") as f:
portfolio/file_utils.py:287:def atomic_write_jsonl(path, entries):
portfolio/file_utils.py:296:        with os.fdopen(fd, "w", encoding="utf-8") as f:
portfolio/file_utils.py:330:        with open(path, "rb") as f:
portfolio/file_utils.py:340:            entry = json.loads(line)
portfolio/file_utils.py:361:        f = open(path, encoding="utf-8")
portfolio/file_utils.py:371:                json.loads(stripped)
portfolio/file_utils.py:382:        with os.fdopen(fd, "w", encoding="utf-8") as f:
portfolio/shadow_registry.py:49:from portfolio.file_utils import atomic_write_json, load_json
portfolio/shadow_registry.py:68:    data = load_json(str(p), default=None)
portfolio/shadow_registry.py:78:    atomic_write_json(str(p), data)
portfolio/config_validator.py:9:from portfolio.file_utils import load_json
portfolio/config_validator.py:20:# ``binance.key`` / ``binance.secret`` was blocking main-loop startup on
portfolio/config_validator.py:66:    config = load_json(CONFIG_FILE)
scripts/fix_agent_dispatcher.py:8:unresolved entries, respects cooldown and kill-switch, and invokes
scripts/fix_agent_dispatcher.py:41:KILL_SWITCH = DATA_DIR / "fix_agent.disabled"
scripts/fix_agent_dispatcher.py:46:# Exponential backoff on consecutive failures.
scripts/fix_agent_dispatcher.py:47:BACKOFF_SCHEDULE_S = [1800, 7200, 43200]  # 30m → 2h → 12h, then disabled
scripts/fix_agent_dispatcher.py:58:AGENT_ALLOWED_TOOLS = "Read,Edit,Bash"
scripts/fix_agent_dispatcher.py:84:    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
scripts/fix_agent_dispatcher.py:89:            entries.append(json.loads(line))
scripts/fix_agent_dispatcher.py:101:    default: a 1-hour blanket cooldown across all categories. Without
scripts/fix_agent_dispatcher.py:102:    that guard, a corrupt state file would clear all cooldowns and let
scripts/fix_agent_dispatcher.py:111:        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
scripts/fix_agent_dispatcher.py:126:                    "blanket cooldown to prevent runaway spawns. Inspect "
scripts/fix_agent_dispatcher.py:133:        # Conservative default: block ALL spawns for 1 hour via the
scripts/fix_agent_dispatcher.py:134:        # global blocked_until_global field. check_gates honours this
scripts/fix_agent_dispatcher.py:139:            "blocked_until_global": (_now() + timedelta(hours=1)).isoformat(),
scripts/fix_agent_dispatcher.py:149:    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
scripts/fix_agent_dispatcher.py:184:    helper the main loop uses (``portfolio.file_utils.atomic_append_jsonl``).
scripts/fix_agent_dispatcher.py:191:    falling back to a non-atomic ``open("a")`` — silent corruption of the
scripts/fix_agent_dispatcher.py:197:        from portfolio.file_utils import atomic_append_jsonl
scripts/fix_agent_dispatcher.py:205:    atomic_append_jsonl(CRITICAL_ERRORS_LOG, entry)
scripts/fix_agent_dispatcher.py:209:# Gating — kill switch, cooldown, recursion
scripts/fix_agent_dispatcher.py:215:    reason: str  # machine-readable short tag, e.g. "cooldown" / "ok"
scripts/fix_agent_dispatcher.py:228:    var taken at run() startup. Passing it explicitly (rather than
scripts/fix_agent_dispatcher.py:236:        return GateDecision(False, "disabled_by_kill_switch")
scripts/fix_agent_dispatcher.py:238:    # Global block — set by _load_state when the state file was corrupt.
scripts/fix_agent_dispatcher.py:239:    global_block = _parse_iso(state.get("blocked_until_global"))
scripts/fix_agent_dispatcher.py:240:    if global_block and global_block > now:
scripts/fix_agent_dispatcher.py:241:        return GateDecision(False, "global_cooldown")
scripts/fix_agent_dispatcher.py:249:    blocked_until = _parse_iso(cat_state.get("blocked_until"))
scripts/fix_agent_dispatcher.py:250:    if blocked_until and blocked_until > now:
scripts/fix_agent_dispatcher.py:251:        return GateDecision(False, "cooldown")
scripts/fix_agent_dispatcher.py:259:    """Bump cooldown + consecutive_failures for the category."""
scripts/fix_agent_dispatcher.py:269:        entry["blocked_until"] = (now + timedelta(seconds=SELF_HEAL_COOLDOWN_S)).isoformat()
scripts/fix_agent_dispatcher.py:276:            # Beyond the schedule: effectively disabled for 10 years. User
scripts/fix_agent_dispatcher.py:279:            entry["blocked_until"] = (now + timedelta(days=3650)).isoformat()
scripts/fix_agent_dispatcher.py:281:            entry["blocked_until"] = (now + timedelta(seconds=BACKOFF_SCHEDULE_S[idx])).isoformat()
scripts/fix_agent_dispatcher.py:333:def run(
scripts/fix_agent_dispatcher.py:402:        # call so the child Claude subprocess inherits depth+1 (blocking any
scripts/fix_agent_dispatcher.py:458:        return run(dry_run=args.dry_run, lookback_h=args.lookback_h)
portfolio/subprocess_utils.py:4:- run_safe(): Drop-in subprocess.run() replacement that uses Windows Job Objects
portfolio/subprocess_utils.py:24:    Drop-in replacement for subprocess.run().  On Windows, creates a Job Object
portfolio/subprocess_utils.py:28:    Falls back to plain subprocess.run() on non-Windows or if Job Object
portfolio/subprocess_utils.py:32:    others accepted by subprocess.Popen / subprocess.run).
portfolio/subprocess_utils.py:35:        return subprocess.run(cmd, **kwargs)
portfolio/subprocess_utils.py:40:        logger.debug("Job Object creation failed (%s), falling back to subprocess.run", exc)
portfolio/subprocess_utils.py:41:        return subprocess.run(cmd, **kwargs)
portfolio/subprocess_utils.py:54:    job = kernel32.CreateJobObjectW(None, None)
portfolio/subprocess_utils.py:56:        raise OSError("CreateJobObjectW failed")
portfolio/subprocess_utils.py:122:            popen_kwargs["stdout"] = subprocess.PIPE
portfolio/subprocess_utils.py:123:            popen_kwargs["stderr"] = subprocess.PIPE
portfolio/subprocess_utils.py:127:            popen_kwargs["stdin"] = subprocess.PIPE
portfolio/subprocess_utils.py:129:        proc = subprocess.Popen(cmd, **popen_kwargs)
portfolio/subprocess_utils.py:132:            kernel32.AssignProcessToJobObject(job, int(proc._handle))
portfolio/subprocess_utils.py:141:        except subprocess.TimeoutExpired:
portfolio/subprocess_utils.py:146:        return subprocess.CompletedProcess(
portfolio/subprocess_utils.py:159:    Like subprocess.Popen(), but assigns the child to a Job Object with
portfolio/subprocess_utils.py:168:    proc = subprocess.Popen(cmd, **kwargs)
portfolio/subprocess_utils.py:175:        kernel32.AssignProcessToJobObject(job, int(proc._handle))
portfolio/subprocess_utils.py:215:        result = subprocess.run(
portfolio/subprocess_utils.py:216:            ["wmic", "process", "where",
portfolio/subprocess_utils.py:239:            subprocess.run(
portfolio/subprocess_utils.py:240:                ["taskkill", "/F", "/PID", str(pid)],
portfolio/subprocess_utils.py:269:        'powershell.exe -NoProfile -Command "'
portfolio/subprocess_utils.py:275:        result = subprocess.run(
portfolio/subprocess_utils.py:280:            shell=True,
portfolio/subprocess_utils.py:290:        data = json.loads(result.stdout.strip())
portfolio/process_lock.py:1:"""Cross-platform helpers for non-blocking singleton process locks."""
portfolio/process_lock.py:22:def acquire_lock_file(
portfolio/process_lock.py:23:    lock_path: str | Path,
portfolio/process_lock.py:28:    """Acquire a non-blocking file lock and return the open handle.
portfolio/process_lock.py:30:    Returns None if another process already holds the lock.
portfolio/process_lock.py:32:    path = Path(lock_path)
portfolio/process_lock.py:36:    fh = path.open("a+", encoding="utf-8")
portfolio/process_lock.py:38:        _lock_file(fh)
portfolio/process_lock.py:43:    _write_lock_metadata(fh, owner=owner, metadata=metadata)
portfolio/process_lock.py:47:def release_lock_file(fh: IO[str] | None) -> None:
portfolio/process_lock.py:48:    """Release a previously acquired lock handle."""
portfolio/process_lock.py:52:        _unlock_file(fh)
portfolio/process_lock.py:60:def _lock_file(fh: IO[str]) -> None:
portfolio/process_lock.py:63:        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
portfolio/process_lock.py:66:        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
portfolio/process_lock.py:69:        "No file locking mechanism available (neither msvcrt nor fcntl). "
portfolio/process_lock.py:74:def _unlock_file(fh: IO[str]) -> None:
portfolio/process_lock.py:77:        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
portfolio/process_lock.py:80:        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
portfolio/process_lock.py:83:def _write_lock_metadata(
portfolio/message_store.py:27:from portfolio.file_utils import atomic_append_jsonl
portfolio/message_store.py:102:    atomic_append_jsonl(MESSAGES_FILE, entry)
portfolio/message_store.py:193:    # Per-category blocklist: mute specific categories
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: scripts/win/*.bat: The filename, directory name, or volume label syntax is incorrect. (os error 123)
rg: scripts/win/*.ps1: The filename, directory name, or volume label syntax is incorrect. (os error 123)
rg: conftest.py: The system cannot find the file specified. (os error 2)

codex
The dashboard surface is mostly read-only, but I still need to verify a few raw file reads and auth edge cases against exact handlers. I’m moving into the dispatcher/checker scripts and Windows task wrappers now, because that’s where the outage-specific invariants live.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg --files | rg "(''^|['"\\\\/])conftest\\.py"'$|conftest"' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "''^" scripts/check_critical_errors.py' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "''^" scripts/fix_agent_dispatcher.py' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "''^" scripts/win' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "''^" dashboard/app.py | Select-String -Pattern "755:|777:|809:|825:|834:|843:|852:|875:|922:|929:|956:|967:|1047:|1055:|1067:|1095:|1110:|1209:|1217:|1243:|1268:|1278:|1309:|1352:|1368:|1385:|1405:|1462:|1493:|1513:|1533:|1551:|1589:|1615:|1639:"' in Q:\finance-analyzer
 succeeded in 719ms:
1:"""Surface unresolved critical errors from data/critical_errors.jsonl.
2:
3:Invoked at the start of every Claude Code session in this project (via
4:the STARTUP CHECK block in CLAUDE.md). Prints a compact summary of
5:unresolved entries from the last 7 days and exits non-zero if any are
6:found. The non-zero exit makes this suitable for future hook wiring.
7:
8:Design notes:
9:
10:* Append-only journal. Resolutions are recorded as follow-up entries with
11:  a ``resolves_ts`` reference rather than mutating earlier entries.
12:* 7-day lookback is arbitrary but long enough to span a weekend + a
13:  trading week. Tune via ``--days N``.
14:* Zero-error output stays silent to avoid adding noise when the system
15:  is healthy; non-zero output is compact enough to fit in a session's
16:  preamble without crowding the user's actual task.
17:"""
18:
19:from __future__ import annotations
20:
21:import argparse
22:import json
23:import sys
24:from datetime import UTC, datetime, timedelta
25:from pathlib import Path
26:
27:DEFAULT_JOURNAL = Path(__file__).resolve().parent.parent / "data" / "critical_errors.jsonl"
28:DEFAULT_DAYS = 7
29:
30:
31:def _parse_ts(ts: str) -> datetime | None:
32:    try:
33:        return datetime.fromisoformat(ts)
34:    except (ValueError, TypeError):
35:        return None
36:
37:
38:def _load_entries(journal: Path) -> list[dict]:
39:    if not journal.exists():
40:        return []
41:    entries = []
42:    for line in journal.read_text(encoding="utf-8", errors="replace").splitlines():
43:        line = line.strip()
44:        if not line:
45:            continue
46:        try:
47:            entries.append(json.loads(line))
48:        except json.JSONDecodeError:
49:            continue
50:    return entries
51:
52:
53:def find_unresolved(entries: list[dict], *, days: int, now: datetime | None = None) -> list[dict]:
54:    """Return entries with resolution=None from the last `days` days.
55:
56:    A later entry with ``resolves_ts`` pointing at an earlier entry's ``ts``
57:    retroactively resolves that earlier entry.
58:    """
59:    now = now or datetime.now(UTC)
60:    cutoff = now - timedelta(days=days)
61:
62:    resolved_ts: set[str] = set()
63:    for e in entries:
64:        rts = e.get("resolves_ts")
65:        if rts:
66:            resolved_ts.add(rts)
67:
68:    unresolved = []
69:    for e in entries:
70:        # Only surface critical-level entries. The fix_agent_dispatcher
71:        # (added 2026-04-13) writes info-level fix_attempt_started /
72:        # fix_attempt_completed lines to the same journal for audit
73:        # purposes; those aren't user-actionable and would create
74:        # rolling noise on every Claude session start.
75:        if e.get("level") != "critical":
76:            continue
77:        if e.get("resolution") is not None:
78:            continue
79:        if e.get("ts") in resolved_ts:
80:            continue
81:        parsed = _parse_ts(e.get("ts", ""))
82:        if parsed is None or parsed < cutoff:
83:            continue
84:        unresolved.append(e)
85:    return unresolved
86:
87:
88:def format_entry(entry: dict) -> str:
89:    ts = entry.get("ts", "?")
90:    category = entry.get("category", "?")
91:    caller = entry.get("caller", "?")
92:    msg = entry.get("message", "")
93:    if len(msg) > 180:
94:        msg = msg[:177] + "..."
95:    return f"[{ts}] {category} caller={caller} :: {msg}"
96:
97:
98:def main(argv: list[str] | None = None) -> int:
99:    # Force UTF-8 stdout: violation messages contain `→` (U+2192) which Windows'
100:    # default cp1252 codec can't encode. Without this, the script crashes mid-print
101:    # and the user sees only the count line, not the entries — meaning unresolved
102:    # accuracy_degradation rows were silently invisible to the STARTUP CHECK
103:    # documented in CLAUDE.md (caught 2026-04-28 during contract-alert diagnosis).
104:    if hasattr(sys.stdout, "reconfigure"):
105:        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
106:
107:    parser = argparse.ArgumentParser(description=__doc__)
108:    parser.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL,
109:                        help="Path to critical_errors.jsonl")
110:    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
111:                        help="Lookback window in days")
112:    parser.add_argument("--json", action="store_true",
113:                        help="Emit raw JSON entries instead of formatted lines")
114:    args = parser.parse_args(argv)
115:
116:    entries = _load_entries(args.journal)
117:    unresolved = find_unresolved(entries, days=args.days)
118:
119:    if not unresolved:
120:        return 0
121:
122:    if args.json:
123:        for e in unresolved:
124:            print(json.dumps(e, ensure_ascii=False))
125:    else:
126:        print(f"{len(unresolved)} unresolved critical error(s) in last {args.days} days:")
127:        print("Journal: " + str(args.journal))
128:        for e in unresolved:
129:            print("  " + format_entry(e))
130:        print()
131:        print("Surface these to the user before continuing. To resolve, append a "
132:              "follow-up entry with resolves_ts set to the original ts.")
133:    return 1
134:
135:
136:if __name__ == "__main__":
137:    sys.exit(main())
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 730ms:
tests\conftest.py
tests\test_avanza_pkg\conftest.py
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 732ms:
1:"""Auto-spawn a Claude Code fix agent when critical errors accumulate.
2:
3:Runs as a separate scheduled task (``PF-FixAgentDispatcher``, every 10m)
4:rather than inline in the main loop — failure of the dispatcher must never
5:destabilise trading.
6:
7:Source of truth: ``data/critical_errors.jsonl``. The dispatcher reads
8:unresolved entries, respects cooldown and kill-switch, and invokes
9:``portfolio.claude_gate.invoke_claude`` with a fix-agent prompt. The
10:agent is instructed to append resolution lines back into the journal;
11:the next dispatcher run sees the resolution and stops re-firing.
12:
13:Design docs: ``docs/plans/2026-04-13-auto-spawn-fix-agent.md``.
14:
15:Exits 0 on every healthy code path — non-zero only on unexpected errors
16:(so the scheduled task's "last result" surfaces real breakage, not
17:routine "nothing to do" runs).
18:"""
19:
20:from __future__ import annotations
21:
22:import argparse
23:import json
24:import logging
25:import os
26:import sys
27:from collections import defaultdict
28:from dataclasses import dataclass
29:from datetime import UTC, datetime, timedelta
30:from pathlib import Path
31:
32:# Ensure Q:\finance-analyzer is importable when invoked from a scheduled task
33:# (which may cwd elsewhere). This mirrors the pattern in other scripts/.
34:BASE_DIR = Path(__file__).resolve().parent.parent
35:if str(BASE_DIR) not in sys.path:
36:    sys.path.insert(0, str(BASE_DIR))
37:
38:DATA_DIR = BASE_DIR / "data"
39:CRITICAL_ERRORS_LOG = DATA_DIR / "critical_errors.jsonl"
40:STATE_FILE = DATA_DIR / "fix_agent_state.json"
41:KILL_SWITCH = DATA_DIR / "fix_agent.disabled"
42:
43:# --- Tunables ---
44:# Same constant the loop contract uses so behavior is coherent.
45:SELF_HEAL_COOLDOWN_S = 1800  # 30 min between attempts per category
46:# Exponential backoff on consecutive failures.
47:BACKOFF_SCHEDULE_S = [1800, 7200, 43200]  # 30m → 2h → 12h, then disabled
48:# Recursion guard: dispatcher must refuse to fire if invoked from within
49:# another fix-agent subprocess (env flag propagates to child Claude).
50:RECURSION_ENV = "PF_FIX_AGENT_DEPTH"
51:MAX_RECURSION_DEPTH = 1
52:# Look-back window for unresolved errors (24h). Older issues are stale.
53:LOOKBACK_H = 24
54:# Per-attempt budget (seconds, Opus 30 turns is typically <15 min).
55:AGENT_TIMEOUT_S = 900
56:AGENT_MAX_TURNS = 30
57:AGENT_MODEL = "opus"
58:AGENT_ALLOWED_TOOLS = "Read,Edit,Bash"
59:
60:logger = logging.getLogger("fix_agent_dispatcher")
61:
62:
63:# ---------------------------------------------------------------------------
64:# Journal I/O — tolerant of malformed / missing files (never raises)
65:# ---------------------------------------------------------------------------
66:
67:def _now() -> datetime:
68:    return datetime.now(UTC)
69:
70:
71:def _parse_iso(ts: str | None) -> datetime | None:
72:    if not ts:
73:        return None
74:    try:
75:        return datetime.fromisoformat(ts)
76:    except (ValueError, TypeError):
77:        return None
78:
79:
80:def _read_journal(path: Path) -> list[dict]:
81:    if not path.exists():
82:        return []
83:    entries = []
84:    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
85:        line = line.strip()
86:        if not line:
87:            continue
88:        try:
89:            entries.append(json.loads(line))
90:        except json.JSONDecodeError:
91:            continue
92:    return entries
93:
94:
95:def _empty_state() -> dict:
96:    return {"by_category": {}, "recursion_counter": 0}
97:
98:
99:def _load_state() -> dict:
100:    """Load dispatcher state. On corruption, returns a *conservative*
101:    default: a 1-hour blanket cooldown across all categories. Without
102:    that guard, a corrupt state file would clear all cooldowns and let
103:    the next dispatcher tick fire every category — including any that
104:    were intentionally backed-off. The conservative default also makes
105:    corruption visible in critical_errors.jsonl rather than silently
106:    accepted.
107:    """
108:    if not STATE_FILE.exists():
109:        return _empty_state()
110:    try:
111:        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
112:    except (OSError, json.JSONDecodeError) as e:
113:        # Record the corruption so the user (and any downstream startup
114:        # check) sees what happened. _append_critical is import-safe so
115:        # this won't cascade if the corruption is part of a broader
116:        # filesystem issue.
117:        try:
118:            _append_critical({
119:                "ts": _now().isoformat(),
120:                "level": "critical",
121:                "category": "fix_agent_state_corrupt",
122:                "caller": "fix_agent_dispatcher",
123:                "resolution": None,
124:                "message": (
125:                    "fix_agent_state.json failed to parse — applying 1h "
126:                    "blanket cooldown to prevent runaway spawns. Inspect "
127:                    "the state file and either fix it or delete it."
128:                ),
129:                "context": {"error": str(e)},
130:            })
131:        except Exception:
132:            pass
133:        # Conservative default: block ALL spawns for 1 hour via the
134:        # global blocked_until_global field. check_gates honours this
135:        # before any per-category logic.
136:        return {
137:            "by_category": {},
138:            "recursion_counter": 0,
139:            "blocked_until_global": (_now() + timedelta(hours=1)).isoformat(),
140:            "_corrupt_loaded_at": _now().isoformat(),
141:        }
142:
143:
144:def _save_state(state: dict) -> None:
145:    """Atomic state-file write — tmp + rename. A mid-write crash must
146:    never leave a corrupt JSON that would break the next dispatcher run."""
147:    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
148:    tmp = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
149:    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
150:    # os.replace is atomic on Windows + POSIX when src/dest are on the same volume.
151:    os.replace(tmp, STATE_FILE)
152:
153:
154:def _find_unresolved(entries: list[dict], lookback_h: int) -> list[dict]:
155:    """Return unresolved entries from the last `lookback_h` hours.
156:
157:    An entry is resolved if (a) it has a non-null ``resolution`` field, or
158:    (b) a later entry has ``resolves_ts`` pointing at its ``ts``.
159:    """
160:    cutoff = _now() - timedelta(hours=lookback_h)
161:    resolved_ts: set[str] = set()
162:    for e in entries:
163:        if e.get("resolves_ts"):
164:            resolved_ts.add(e["resolves_ts"])
165:
166:    unresolved = []
167:    for e in entries:
168:        # Only treat "critical" level entries as actionable; skip info/resolution lines
169:        if e.get("level") != "critical":
170:            continue
171:        if e.get("resolution") is not None:
172:            continue
173:        if e.get("ts") in resolved_ts:
174:            continue
175:        parsed = _parse_iso(e.get("ts"))
176:        if parsed is None or parsed < cutoff:
177:            continue
178:        unresolved.append(e)
179:    return unresolved
180:
181:
182:def _append_critical(entry: dict) -> None:
183:    """Append a record to critical_errors.jsonl via the same atomic
184:    helper the main loop uses (``portfolio.file_utils.atomic_append_jsonl``).
185:
186:    The main loop's ``claude_gate.record_critical_error`` writes to this
187:    file concurrently from multiple processes — using the same primitive
188:    here keeps interleaving semantics consistent.
189:
190:    If the import fails (broken install), we LOG and SKIP rather than
191:    falling back to a non-atomic ``open("a")`` — silent corruption of the
192:    journal that surfaces critical errors to every Claude session would
193:    be worse than dropping a single dispatcher record.
194:    """
195:    CRITICAL_ERRORS_LOG.parent.mkdir(parents=True, exist_ok=True)
196:    try:
197:        from portfolio.file_utils import atomic_append_jsonl
198:    except ImportError:
199:        logger.error(
200:            "portfolio.file_utils unavailable — dispatcher cannot safely "
201:            "append to critical_errors.jsonl. Skipping record: %r",
202:            entry.get("category"),
203:        )
204:        return
205:    atomic_append_jsonl(CRITICAL_ERRORS_LOG, entry)
206:
207:
208:# ---------------------------------------------------------------------------
209:# Gating — kill switch, cooldown, recursion
210:# ---------------------------------------------------------------------------
211:
212:@dataclass
213:class GateDecision:
214:    allowed: bool
215:    reason: str  # machine-readable short tag, e.g. "cooldown" / "ok"
216:
217:
218:def check_gates(
219:    category: str,
220:    state: dict,
221:    now: datetime | None = None,
222:    *,
223:    caller_depth: int | None = None,
224:) -> GateDecision:
225:    """Decide whether a fix attempt for *category* is permitted right now.
226:
227:    ``caller_depth`` is the dispatcher's snapshot of the recursion env
228:    var taken at run() startup. Passing it explicitly (rather than
229:    re-reading os.environ here) keeps sibling categories from being
230:    treated as recursion when the dispatcher mutates os.environ around
231:    each invoke_claude call.
232:    """
233:    now = now or _now()
234:
235:    if KILL_SWITCH.exists():
236:        return GateDecision(False, "disabled_by_kill_switch")
237:
238:    # Global block — set by _load_state when the state file was corrupt.
239:    global_block = _parse_iso(state.get("blocked_until_global"))
240:    if global_block and global_block > now:
241:        return GateDecision(False, "global_cooldown")
242:
243:    if caller_depth is None:
244:        caller_depth = int(os.environ.get(RECURSION_ENV, "0") or "0")
245:    if caller_depth >= MAX_RECURSION_DEPTH:
246:        return GateDecision(False, "recursion_depth_exceeded")
247:
248:    cat_state = state.get("by_category", {}).get(category, {})
249:    blocked_until = _parse_iso(cat_state.get("blocked_until"))
250:    if blocked_until and blocked_until > now:
251:        return GateDecision(False, "cooldown")
252:
253:    return GateDecision(True, "ok")
254:
255:
256:def update_state_after_attempt(
257:    state: dict, category: str, success: bool, now: datetime | None = None,
258:) -> dict:
259:    """Bump cooldown + consecutive_failures for the category."""
260:    now = now or _now()
261:    cats = state.setdefault("by_category", {})
262:    entry = cats.setdefault(category, {"consecutive_failures": 0})
263:
264:    entry["last_attempt_ts"] = now.isoformat()
265:    entry["last_attempt_success"] = success
266:
267:    if success:
268:        entry["consecutive_failures"] = 0
269:        entry["blocked_until"] = (now + timedelta(seconds=SELF_HEAL_COOLDOWN_S)).isoformat()
270:    else:
271:        prev = entry.get("consecutive_failures", 0)
272:        new_count = prev + 1
273:        entry["consecutive_failures"] = new_count
274:        idx = min(new_count - 1, len(BACKOFF_SCHEDULE_S) - 1)
275:        if new_count > len(BACKOFF_SCHEDULE_S):
276:            # Beyond the schedule: effectively disabled for 10 years. User
277:            # must manually reset by editing state file or adding a
278:            # resolution line.
279:            entry["blocked_until"] = (now + timedelta(days=3650)).isoformat()
280:        else:
281:            entry["blocked_until"] = (now + timedelta(seconds=BACKOFF_SCHEDULE_S[idx])).isoformat()
282:    return state
283:
284:
285:# ---------------------------------------------------------------------------
286:# Prompt construction
287:# ---------------------------------------------------------------------------
288:
289:def build_fix_prompt(category: str, entries: list[dict]) -> str:
290:    """Context-complete prompt for the fix agent — the agent runs in a
291:    fresh conversation and has no memory of how the errors were recorded."""
292:    bullet_points = []
293:    for e in entries[:10]:  # cap at 10 to bound prompt size
294:        bullet_points.append(
295:            f"- [{e.get('ts','?')}] {e.get('category','?')} "
296:            f"caller={e.get('caller','?')} :: {e.get('message','')}"
297:        )
298:    bullets = "\n".join(bullet_points) or "(no details available)"
299:
300:    return (
301:        "You are the Layer 2 fix agent for finance-analyzer. A critical error "
302:        "was recorded in data/critical_errors.jsonl and has not been "
303:        "auto-resolved. Your job is to diagnose and either fix it or document "
304:        "why it requires human attention.\n\n"
305:        f"## Unresolved critical errors (category: {category})\n\n"
306:        f"{bullets}\n\n"
307:        "## Your instructions\n\n"
308:        "1. Read CLAUDE.md and any source files referenced by the error messages.\n"
309:        "2. Identify the root cause.\n"
310:        "3. Either:\n"
311:        "   a. Make the fix directly using Edit (preferred for simple regressions).\n"
312:        "   b. Write a fix proposal to data/proposed_fixes/<timestamp>.md when the\n"
313:        "      fix is risky, out of scope, or requires user decisions.\n"
314:        "4. When done, append a resolution line to data/critical_errors.jsonl:\n"
315:        '   {"ts":"<ISO UTC now>","level":"info","category":"resolution",\n'
316:        '    "caller":"fix_agent","resolves_ts":"<original ts>",\n'
317:        '    "resolution":"<short description>","message":"<details>","context":{}}\n\n'
318:        "DO NOT:\n"
319:        "- Modify files outside portfolio/, scripts/, tests/, docs/.\n"
320:        "- Kill processes or restart any loop.\n"
321:        "- Edit config.json, .env, or anything in ~/.claude.\n"
322:        "- Commit or push.\n\n"
323:        "If you cannot safely fix it, still append a resolution line explaining\n"
324:        "what you investigated and why the fix requires human action — this\n"
325:        "stops the dispatcher re-firing on the same error.\n"
326:    )
327:
328:
329:# ---------------------------------------------------------------------------
330:# Main dispatch
331:# ---------------------------------------------------------------------------
332:
333:def run(
334:    dry_run: bool = False,
335:    lookback_h: int = LOOKBACK_H,
336:    invoke_claude_fn=None,
337:) -> int:
338:    """Dispatcher entry point. Returns 0 on success (including no-op).
339:
340:    ``invoke_claude_fn`` is dependency-injected for tests; production
341:    passes None and we import at call time.
342:    """
343:    entries = _read_journal(CRITICAL_ERRORS_LOG)
344:    unresolved = _find_unresolved(entries, lookback_h)
345:    if not unresolved:
346:        logger.info("No unresolved critical errors — exiting")
347:        return 0
348:
349:    # Group by category so one agent handles related entries
350:    by_category: dict[str, list[dict]] = defaultdict(list)
351:    for e in unresolved:
352:        by_category[e.get("category", "unknown")].append(e)
353:
354:    state = _load_state()
355:    any_spawned = False
356:
357:    # Snapshot the recursion depth ONCE at startup. Each category's spawn
358:    # temporarily bumps os.environ[RECURSION_ENV] so the child Claude
359:    # subprocess inherits depth+1, but we reset afterwards so successive
360:    # categories in the same dispatcher run are treated as siblings, not
361:    # recursion.
362:    caller_recursion_depth = int(os.environ.get(RECURSION_ENV, "0") or "0")
363:
364:    for category, cat_entries in by_category.items():
365:        decision = check_gates(category, state, caller_depth=caller_recursion_depth)
366:        if not decision.allowed:
367:            logger.info("Skipping category=%s (%s)", category, decision.reason)
368:            _append_critical({
369:                "ts": _now().isoformat(),
370:                "level": "info",
371:                "category": "fix_attempt_skipped",
372:                "caller": "fix_agent_dispatcher",
373:                "resolution": None,
374:                "message": f"Skipped fix attempt for {category}: {decision.reason}",
375:                "context": {"skipped_category": category, "reason": decision.reason,
376:                            "unresolved_count": len(cat_entries)},
377:            })
378:            continue
379:
380:        _append_critical({
381:            "ts": _now().isoformat(),
382:            "level": "info",
383:            "category": "fix_attempt_started",
384:            "caller": "fix_agent_dispatcher",
385:            "resolution": None,
386:            "message": f"Spawning fix agent for category={category}",
387:            "context": {"target_category": category, "entry_count": len(cat_entries),
388:                        "dry_run": dry_run},
389:        })
390:
391:        if dry_run:
392:            logger.info("DRY RUN — would spawn fix agent for category=%s", category)
393:            any_spawned = True
394:            continue
395:
396:        prompt = build_fix_prompt(category, cat_entries)
397:
398:        if invoke_claude_fn is None:
399:            from portfolio.claude_gate import invoke_claude as invoke_claude_fn  # type: ignore
400:
401:        # Set the recursion env flag only for the duration of the invoke_claude
402:        # call so the child Claude subprocess inherits depth+1 (blocking any
403:        # transitive dispatcher re-entry). Restore afterwards so subsequent
404:        # categories in this same run aren't mistaken for recursion.
405:        prior_env = os.environ.get(RECURSION_ENV)
406:        os.environ[RECURSION_ENV] = str(caller_recursion_depth + 1)
407:        try:
408:            success, exit_code = invoke_claude_fn(
409:                prompt=prompt,
410:                caller=f"fix_agent_{category}",
411:                model=AGENT_MODEL,
412:                max_turns=AGENT_MAX_TURNS,
413:                allowed_tools=AGENT_ALLOWED_TOOLS,
414:                timeout=AGENT_TIMEOUT_S,
415:            )
416:        except Exception as e:
417:            logger.exception("Fix agent invocation raised: %s", e)
418:            success, exit_code = False, -1
419:        finally:
420:            if prior_env is None:
421:                os.environ.pop(RECURSION_ENV, None)
422:            else:
423:                os.environ[RECURSION_ENV] = prior_env
424:
425:        any_spawned = True
426:        state = update_state_after_attempt(state, category, success)
427:        _append_critical({
428:            "ts": _now().isoformat(),
429:            "level": "info" if success else "critical",
430:            "category": "fix_attempt_completed" if success else "fix_agent_failed",
431:            "caller": "fix_agent_dispatcher",
432:            "resolution": None,  # agent will write its own resolution line
433:            "message": (
434:                f"Fix agent for {category} {'succeeded' if success else 'FAILED'} "
435:                f"(exit={exit_code}). Check journal for the agent's resolution line."
436:            ),
437:            "context": {"target_category": category, "success": success,
438:                        "exit_code": exit_code,
439:                        "consecutive_failures":
440:                            state["by_category"][category]["consecutive_failures"]},
441:        })
442:
443:    _save_state(state)
444:    logger.info("Dispatcher run complete (spawned=%s)", any_spawned)
445:    return 0
446:
447:
448:def main(argv: list[str] | None = None) -> int:
449:    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
450:    parser = argparse.ArgumentParser(description=__doc__)
451:    parser.add_argument("--dry-run", action="store_true",
452:                        help="Don't actually spawn the agent — just log what would happen")
453:    parser.add_argument("--lookback-h", type=int, default=LOOKBACK_H,
454:                        help=f"Hours of critical_errors.jsonl to inspect (default {LOOKBACK_H})")
455:    args = parser.parse_args(argv)
456:
457:    try:
458:        return run(dry_run=args.dry_run, lookback_h=args.lookback_h)
459:    except Exception as e:
460:        logger.exception("Dispatcher crashed: %s", e)
461:        # Leave an explicit marker so the user knows the dispatcher itself
462:        # broke — not just the underlying fix attempts.
463:        try:
464:            _append_critical({
465:                "ts": _now().isoformat(),
466:                "level": "critical",
467:                "category": "fix_agent_dispatcher_crashed",
468:                "caller": "fix_agent_dispatcher",
469:                "resolution": None,
470:                "message": f"Dispatcher crashed: {e}",
471:                "context": {"exception_type": type(e).__name__},
472:            })
473:        except Exception:
474:            pass
475:        return 2
476:
477:
478:if __name__ == "__main__":
479:    sys.exit(main())
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 732ms:
scripts/win\fin-snipe-manager.bat:1:@echo off
scripts/win\fin-snipe-manager.bat:2:cd /d %~dp0\..\..
scripts/win\fin-snipe-manager.bat:3:.venv\Scripts\python.exe -u -m portfolio.fin_snipe_manager %*
scripts/win\ft-download-data.bat:1:@echo off
scripts/win\ft-download-data.bat:2:REM Download historical OHLCV data for backtesting.
scripts/win\ft-download-data.bat:3:REM Usage: ft-download-data.bat [days] [timeframes...]
scripts/win\ft-download-data.bat:4:REM Example: ft-download-data.bat 730 5m 1h 4h 1d
scripts/win\ft-download-data.bat:5:
scripts/win\ft-download-data.bat:6:setlocal enabledelayedexpansion
scripts/win\ft-download-data.bat:7:set "PROJECT_DIR=%~dp0..\.."
scripts/win\ft-download-data.bat:8:call "%PROJECT_DIR%\.venv\Scripts\activate.bat"
scripts/win\ft-download-data.bat:9:
scripts/win\ft-download-data.bat:10:set "DAYS=%1"
scripts/win\ft-download-data.bat:11:if "%DAYS%"=="" set "DAYS=30"
scripts/win\ft-download-data.bat:12:shift
scripts/win\ft-download-data.bat:13:
scripts/win\ft-download-data.bat:14:set "HAS_TF=0"
scripts/win\ft-download-data.bat:15::loop
scripts/win\ft-download-data.bat:16:if "%1"=="" goto :done
scripts/win\ft-download-data.bat:17:set "HAS_TF=1"
scripts/win\ft-download-data.bat:18:echo ==^> Downloading %DAYS%d of %1 data...
scripts/win\ft-download-data.bat:19:freqtrade download-data ^
scripts/win\ft-download-data.bat:20:    --config "%PROJECT_DIR%\config.json" ^
scripts/win\ft-download-data.bat:21:    --timeframe %1 ^
scripts/win\ft-download-data.bat:22:    --days %DAYS%
scripts/win\ft-download-data.bat:23:shift
scripts/win\ft-download-data.bat:24:goto :loop
scripts/win\ft-download-data.bat:25:
scripts/win\ft-download-data.bat:26::done
scripts/win\ft-download-data.bat:27:if "%HAS_TF%"=="0" (
scripts/win\ft-download-data.bat:28:    echo ==^> Downloading %DAYS%d of 5m data...
scripts/win\ft-download-data.bat:29:    freqtrade download-data ^
scripts/win\ft-download-data.bat:30:        --config "%PROJECT_DIR%\config.json" ^
scripts/win\ft-download-data.bat:31:        --timeframe 5m ^
scripts/win\ft-download-data.bat:32:        --days %DAYS%
scripts/win\ft-download-data.bat:33:)
scripts/win\ft-dry-run.bat:1:@echo off
scripts/win\ft-dry-run.bat:2:REM Start paper trading (dry run).
scripts/win\ft-dry-run.bat:3:REM Usage: ft-dry-run.bat [extra args...]
scripts/win\ft-dry-run.bat:4:
scripts/win\ft-dry-run.bat:5:setlocal
scripts/win\ft-dry-run.bat:6:set "PROJECT_DIR=%~dp0..\.."
scripts/win\ft-dry-run.bat:7:call "%PROJECT_DIR%\.venv\Scripts\activate.bat"
scripts/win\ft-dry-run.bat:8:
scripts/win\ft-dry-run.bat:9:if exist "%PROJECT_DIR%\config.json" (
scripts/win\ft-dry-run.bat:10:    set "CONFIG=%PROJECT_DIR%\config.json"
scripts/win\ft-dry-run.bat:11:) else (
scripts/win\ft-dry-run.bat:12:    set "CONFIG=%PROJECT_DIR%\config.example.json"
scripts/win\ft-dry-run.bat:13:)
scripts/win\ft-dry-run.bat:14:
scripts/win\ft-dry-run.bat:15:freqtrade trade ^
scripts/win\ft-dry-run.bat:16:    --config "%CONFIG%" ^
scripts/win\ft-dry-run.bat:17:    --strategy TABaseStrategy ^
scripts/win\ft-dry-run.bat:18:    --strategy-path "%PROJECT_DIR%\user_data\strategies" ^
scripts/win\ft-dry-run.bat:19:    %*
scripts/win\ft-hyperopt.bat:1:@echo off
scripts/win\ft-hyperopt.bat:2:REM Run hyperparameter optimization for TABaseStrategy.
scripts/win\ft-hyperopt.bat:3:REM Usage: ft-hyperopt.bat [epochs] [extra args...]
scripts/win\ft-hyperopt.bat:4:REM Example: ft-hyperopt.bat 500 --timerange 20240101-
scripts/win\ft-hyperopt.bat:5:
scripts/win\ft-hyperopt.bat:6:setlocal
scripts/win\ft-hyperopt.bat:7:set "PROJECT_DIR=%~dp0..\.."
scripts/win\ft-hyperopt.bat:8:call "%PROJECT_DIR%\.venv\Scripts\activate.bat"
scripts/win\ft-hyperopt.bat:9:
scripts/win\ft-hyperopt.bat:10:set "EPOCHS=%1"
scripts/win\ft-hyperopt.bat:11:if "%EPOCHS%"=="" set "EPOCHS=100"
scripts/win\ft-hyperopt.bat:12:shift
scripts/win\ft-hyperopt.bat:13:
scripts/win\ft-hyperopt.bat:14:freqtrade hyperopt ^
scripts/win\ft-hyperopt.bat:15:    --config "%PROJECT_DIR%\config.json" ^
scripts/win\ft-hyperopt.bat:16:    --strategy TABaseStrategy ^
scripts/win\ft-hyperopt.bat:17:    --strategy-path "%PROJECT_DIR%\user_data\strategies" ^
scripts/win\ft-hyperopt.bat:18:    --hyperopt-loss SharpeHyperOptLossDaily ^
scripts/win\ft-hyperopt.bat:19:    --spaces buy sell ^
scripts/win\ft-hyperopt.bat:20:    --epochs %EPOCHS% ^
scripts/win\ft-hyperopt.bat:21:    %1 %2 %3 %4 %5 %6 %7 %8 %9
scripts/win\ft-backtest.bat:1:@echo off
scripts/win\ft-backtest.bat:2:REM Run backtests with TABaseStrategy.
scripts/win\ft-backtest.bat:3:REM Usage: ft-backtest.bat [extra args...]
scripts/win\ft-backtest.bat:4:REM Example: ft-backtest.bat --timerange 20260101-
scripts/win\ft-backtest.bat:5:
scripts/win\ft-backtest.bat:6:setlocal
scripts/win\ft-backtest.bat:7:set "PROJECT_DIR=%~dp0..\.."
scripts/win\ft-backtest.bat:8:call "%PROJECT_DIR%\.venv\Scripts\activate.bat"
scripts/win\ft-backtest.bat:9:
scripts/win\ft-backtest.bat:10:freqtrade backtesting ^
scripts/win\ft-backtest.bat:11:    --config "%PROJECT_DIR%\config.json" ^
scripts/win\ft-backtest.bat:12:    --strategy TABaseStrategy ^
scripts/win\ft-backtest.bat:13:    --strategy-path "%PROJECT_DIR%\user_data\strategies" ^
scripts/win\ft-backtest.bat:14:    %*
scripts/win\install-fix-agent-task.ps1:1:# install-fix-agent-task.ps1 — Run as Administrator
scripts/win\install-fix-agent-task.ps1:2:# Creates PF-FixAgentDispatcher: fires every 10 minutes, runs
scripts/win\install-fix-agent-task.ps1:3:# scripts/fix_agent_dispatcher.py. The dispatcher is a no-op when
scripts/win\install-fix-agent-task.ps1:4:# data/critical_errors.jsonl has no unresolved entries, so firing
scripts/win\install-fix-agent-task.ps1:5:# frequently is cheap. See docs/plans/2026-04-13-auto-spawn-fix-agent.md.
scripts/win\install-fix-agent-task.ps1:6:
scripts/win\install-fix-agent-task.ps1:7:$taskName    = "PF-FixAgentDispatcher"
scripts/win\install-fix-agent-task.ps1:8:$pythonPath  = "Q:\finance-analyzer\.venv\Scripts\python.exe"
scripts/win\install-fix-agent-task.ps1:9:$scriptPath  = "Q:\finance-analyzer\scripts\fix_agent_dispatcher.py"
scripts/win\install-fix-agent-task.ps1:10:$workingDir  = "Q:\finance-analyzer"
scripts/win\install-fix-agent-task.ps1:11:
scripts/win\install-fix-agent-task.ps1:12:# Remove existing task if present (idempotent install)
scripts/win\install-fix-agent-task.ps1:13:Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
scripts/win\install-fix-agent-task.ps1:14:
scripts/win\install-fix-agent-task.ps1:15:# Trigger: every 10 minutes, indefinitely
scripts/win\install-fix-agent-task.ps1:16:$trigger = New-ScheduledTaskTrigger `
scripts/win\install-fix-agent-task.ps1:17:    -Once `
scripts/win\install-fix-agent-task.ps1:18:    -At (Get-Date) `
scripts/win\install-fix-agent-task.ps1:19:    -RepetitionInterval (New-TimeSpan -Minutes 10) `
scripts/win\install-fix-agent-task.ps1:20:    -RepetitionDuration ([TimeSpan]::MaxValue)
scripts/win\install-fix-agent-task.ps1:21:
scripts/win\install-fix-agent-task.ps1:22:# Action: python scripts/fix_agent_dispatcher.py
scripts/win\install-fix-agent-task.ps1:23:$action = New-ScheduledTaskAction `
scripts/win\install-fix-agent-task.ps1:24:    -Execute $pythonPath `
scripts/win\install-fix-agent-task.ps1:25:    -Argument "-u `"$scriptPath`"" `
scripts/win\install-fix-agent-task.ps1:26:    -WorkingDirectory $workingDir
scripts/win\install-fix-agent-task.ps1:27:
scripts/win\install-fix-agent-task.ps1:28:# Settings: cap runtime at 20 minutes (agent timeout is 15 min + buffer).
scripts/win\install-fix-agent-task.ps1:29:# Don't run if battery, skip if a previous instance is still running (
scripts/win\install-fix-agent-task.ps1:30:# MultipleInstances=IgnoreNew), start when available after wake.
scripts/win\install-fix-agent-task.ps1:31:$settings = New-ScheduledTaskSettingsSet `
scripts/win\install-fix-agent-task.ps1:32:    -AllowStartIfOnBatteries `
scripts/win\install-fix-agent-task.ps1:33:    -DontStopIfGoingOnBatteries `
scripts/win\install-fix-agent-task.ps1:34:    -StartWhenAvailable `
scripts/win\install-fix-agent-task.ps1:35:    -ExecutionTimeLimit (New-TimeSpan -Minutes 20) `
scripts/win\install-fix-agent-task.ps1:36:    -MultipleInstances IgnoreNew
scripts/win\install-fix-agent-task.ps1:37:
scripts/win\install-fix-agent-task.ps1:38:$principal = New-ScheduledTaskPrincipal `
scripts/win\install-fix-agent-task.ps1:39:    -UserId $env:USERNAME `
scripts/win\install-fix-agent-task.ps1:40:    -LogonType Interactive
scripts/win\install-fix-agent-task.ps1:41:
scripts/win\install-fix-agent-task.ps1:42:Register-ScheduledTask `
scripts/win\install-fix-agent-task.ps1:43:    -TaskName $taskName `
scripts/win\install-fix-agent-task.ps1:44:    -Action $action `
scripts/win\install-fix-agent-task.ps1:45:    -Trigger $trigger `
scripts/win\install-fix-agent-task.ps1:46:    -Settings $settings `
scripts/win\install-fix-agent-task.ps1:47:    -Principal $principal `
scripts/win\install-fix-agent-task.ps1:48:    -Description "Auto-spawn a Claude fix agent when data/critical_errors.jsonl has unresolved entries. Kill switch: touch data/fix_agent.disabled." `
scripts/win\install-fix-agent-task.ps1:49:    | Out-Null
scripts/win\install-fix-agent-task.ps1:50:
scripts/win\install-fix-agent-task.ps1:51:Write-Host ""
scripts/win\install-fix-agent-task.ps1:52:Write-Host "=== $taskName installed ==="
scripts/win\install-fix-agent-task.ps1:53:Write-Host "Every 10 minutes: $pythonPath -u $scriptPath"
scripts/win\install-fix-agent-task.ps1:54:Write-Host "Working dir:       $workingDir"
scripts/win\install-fix-agent-task.ps1:55:Write-Host "Kill switch:       touch Q:\finance-analyzer\data\fix_agent.disabled"
scripts/win\install-fix-agent-task.ps1:56:Write-Host ""
scripts/win\install-fix-agent-task.ps1:57:Write-Host "To verify: schtasks /Query /TN '$taskName' /V /FO LIST"
scripts/win\install-fix-agent-task.ps1:58:Write-Host "To remove: Unregister-ScheduledTask -TaskName '$taskName' -Confirm:`$false"
scripts/win\add-cloudflared-path.ps1:1:$cfPath = "C:\Program Files (x86)\cloudflared"
scripts/win\add-cloudflared-path.ps1:2:$currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
scripts/win\add-cloudflared-path.ps1:3:if ($currentPath -notlike "*cloudflared*") {
scripts/win\add-cloudflared-path.ps1:4:    [Environment]::SetEnvironmentVariable("PATH", "$currentPath;$cfPath", "Machine")
scripts/win\add-cloudflared-path.ps1:5:    Write-Output "Added $cfPath to system PATH"
scripts/win\add-cloudflared-path.ps1:6:} else {
scripts/win\add-cloudflared-path.ps1:7:    Write-Output "cloudflared already in system PATH"
scripts/win\add-cloudflared-path.ps1:8:}
scripts/win\crypto-loop.bat:1:@echo off
scripts/win\crypto-loop.bat:2:REM Crypto Intraday Trading Loop — BTC + ETH paper-mode swing subsystem.
scripts/win\crypto-loop.bat:3:REM Auto-restarts on crash with 30s delay. Exit code 11 means another
scripts/win\crypto-loop.bat:4:REM instance already holds the singleton lock — we stop instead of
scripts/win\crypto-loop.bat:5:REM fork-bombing into the live instance.
scripts/win\crypto-loop.bat:6:cd /d Q:\finance-analyzer
scripts/win\crypto-loop.bat:7:
scripts/win\crypto-loop.bat:8::restart
scripts/win\crypto-loop.bat:9:REM Clear Claude Code session markers so any subagent invocation can launch.
scripts/win\crypto-loop.bat:10:set CLAUDECODE=
scripts/win\crypto-loop.bat:11:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\crypto-loop.bat:12:echo [%date% %time%] Starting crypto loop...
scripts/win\crypto-loop.bat:13:.venv\Scripts\python.exe -u data\crypto_loop.py --loop > data\crypto_loop_out.txt 2>&1
scripts/win\crypto-loop.bat:14:set EXIT_CODE=%ERRORLEVEL%
scripts/win\crypto-loop.bat:15:echo [%date% %time%] Crypto loop exited (code %EXIT_CODE%).
scripts/win\crypto-loop.bat:16:
scripts/win\crypto-loop.bat:17:REM Duplicate instance detected -- do not loop-restart into the active loop
scripts/win\crypto-loop.bat:18:if %EXIT_CODE% EQU 11 (
scripts/win\crypto-loop.bat:19:    echo [%date% %time%] Another crypto loop instance already holds the lock -- stopping wrapper.
scripts/win\crypto-loop.bat:20:    goto :eof
scripts/win\crypto-loop.bat:21:)
scripts/win\crypto-loop.bat:22:
scripts/win\crypto-loop.bat:23:echo [%date% %time%] Restarting in 30s...
scripts/win\crypto-loop.bat:24:timeout /t 30 /nobreak >nul
scripts/win\crypto-loop.bat:25:goto restart
scripts/win\install-local-llm-report-task.ps1:1:param(
scripts/win\install-local-llm-report-task.ps1:2:    [string]$TaskName = "PF-LocalLlmReport",
scripts/win\install-local-llm-report-task.ps1:3:    [string]$Time = "18:10",
scripts/win\install-local-llm-report-task.ps1:4:    [int]$Days = 30,
scripts/win\install-local-llm-report-task.ps1:5:    [switch]$Remove
scripts/win\install-local-llm-report-task.ps1:6:)
scripts/win\install-local-llm-report-task.ps1:7:
scripts/win\install-local-llm-report-task.ps1:8:$ErrorActionPreference = "Stop"
scripts/win\install-local-llm-report-task.ps1:9:
scripts/win\install-local-llm-report-task.ps1:10:if ($Time -notmatch '^\d{2}:\d{2}$') {
scripts/win\install-local-llm-report-task.ps1:11:    throw "Time must be in HH:mm format."
scripts/win\install-local-llm-report-task.ps1:12:}
scripts/win\install-local-llm-report-task.ps1:13:
scripts/win\install-local-llm-report-task.ps1:14:if ($Days -lt 1) {
scripts/win\install-local-llm-report-task.ps1:15:    throw "Days must be >= 1."
scripts/win\install-local-llm-report-task.ps1:16:}
scripts/win\install-local-llm-report-task.ps1:17:
scripts/win\install-local-llm-report-task.ps1:18:$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
scripts/win\install-local-llm-report-task.ps1:19:$runner = Join-Path $repoRoot "scripts\win\pf-local-llm-report.bat"
scripts/win\install-local-llm-report-task.ps1:20:
scripts/win\install-local-llm-report-task.ps1:21:if (-not (Test-Path $runner)) {
scripts/win\install-local-llm-report-task.ps1:22:    throw "Runner not found: $runner"
scripts/win\install-local-llm-report-task.ps1:23:}
scripts/win\install-local-llm-report-task.ps1:24:
scripts/win\install-local-llm-report-task.ps1:25:if ($Remove) {
scripts/win\install-local-llm-report-task.ps1:26:    & schtasks /Delete /TN $TaskName /F
scripts/win\install-local-llm-report-task.ps1:27:    exit $LASTEXITCODE
scripts/win\install-local-llm-report-task.ps1:28:}
scripts/win\install-local-llm-report-task.ps1:29:
scripts/win\install-local-llm-report-task.ps1:30:$taskCommand = '"' + $runner + '" ' + $Days
scripts/win\install-local-llm-report-task.ps1:31:
scripts/win\install-local-llm-report-task.ps1:32:& schtasks /Create /TN $TaskName /SC DAILY /ST $Time /TR $taskCommand /F
scripts/win\install-local-llm-report-task.ps1:33:if ($LASTEXITCODE -ne 0) {
scripts/win\install-local-llm-report-task.ps1:34:    exit $LASTEXITCODE
scripts/win\install-local-llm-report-task.ps1:35:}
scripts/win\install-local-llm-report-task.ps1:36:
scripts/win\install-local-llm-report-task.ps1:37:Write-Host "Created or updated task: $TaskName"
scripts/win\install-local-llm-report-task.ps1:38:Write-Host "Schedule: daily at $Time"
scripts/win\install-local-llm-report-task.ps1:39:Write-Host "Command: $taskCommand"
scripts/win\fin-snipe.bat:1:@echo off
scripts/win\fin-snipe.bat:2:cd /d %~dp0\..\..
scripts/win\fin-snipe.bat:3:.venv\Scripts\python.exe -u -m portfolio.fin_snipe %*
scripts/win\adversarial-review.bat:1:@echo off
scripts/win\adversarial-review.bat:2:REM PF-AdversarialReview — Daily dual adversarial review (Codex + Claude)
scripts/win\adversarial-review.bat:3:REM Runs claude code CLI with the full review prompt.
scripts/win\adversarial-review.bat:4:REM Output: data\adversarial_review_out.txt
scripts/win\adversarial-review.bat:5:cd /d Q:\finance-analyzer
scripts/win\adversarial-review.bat:6:set CLAUDECODE=
scripts/win\adversarial-review.bat:7:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\adversarial-review.bat:8:
scripts/win\adversarial-review.bat:9:echo [%date% %time%] Starting adversarial review... >> data\adversarial_review_out.txt 2>&1
scripts/win\adversarial-review.bat:10:
scripts/win\adversarial-review.bat:11:claude -p "Follow /fgl protocol. Run a full dual adversarial review of the finance-analyzer codebase: partition into 8 subsystems (signals-core, orchestration, portfolio-risk, metals-core, avanza-api, signals-modules, data-external, infrastructure), create worktree with empty-baseline branches, run /codex:adversarial-review in background for each subsystem, write your own independent adversarial review, collect codex results, cross-critique in both directions, write synthesis doc. Commit all docs to main, push via Windows git, clean up worktrees. Do NOT ask for approval — follow /fgl rules. Spend your entire context on this." --allowedTools "Edit,Read,Bash,Write,Glob,Grep" --max-turns 80 >> data\adversarial_review_out.txt 2>&1
scripts/win\adversarial-review.bat:12:
scripts/win\adversarial-review.bat:13:echo [%date% %time%] Adversarial review finished (code %ERRORLEVEL%). >> data\adversarial_review_out.txt 2>&1
scripts/win\install-crypto-loop-task.ps1:1:# Install the canonical scheduled task for the crypto (BTC+ETH) loop.
scripts/win\install-crypto-loop-task.ps1:2:# Run as:
scripts/win\install-crypto-loop-task.ps1:3:#   powershell -ExecutionPolicy Bypass -File scripts\win\install-crypto-loop-task.ps1
scripts/win\install-crypto-loop-task.ps1:4:#
scripts/win\install-crypto-loop-task.ps1:5:# Parity with PF-MetalsLoop:
scripts/win\install-crypto-loop-task.ps1:6:#   - AtLogOn + weekly Mon-Fri 07:00 triggers
scripts/win\install-crypto-loop-task.ps1:7:#   - 3-day execution time limit (the wrapper auto-restarts on crash)
scripts/win\install-crypto-loop-task.ps1:8:#   - Multiple-instance ignored (singleton lock at the Python level)
scripts/win\install-crypto-loop-task.ps1:9:#   - 3 restarts with 1-min interval on hard failures
scripts/win\install-crypto-loop-task.ps1:10:#
scripts/win\install-crypto-loop-task.ps1:11:# This script DOES NOT auto-start the task. After install, the user runs:
scripts/win\install-crypto-loop-task.ps1:12:#   Start-ScheduledTask -TaskName 'PF-CryptoLoop'
scripts/win\install-crypto-loop-task.ps1:13:# Until then, the loop is registered but inert (paper-mode means zero
scripts/win\install-crypto-loop-task.ps1:14:# trading risk anyway, but explicit user action is the canonical pattern).
scripts/win\install-crypto-loop-task.ps1:15:
scripts/win\install-crypto-loop-task.ps1:16:$TaskName = "PF-CryptoLoop"
scripts/win\install-crypto-loop-task.ps1:17:$scriptDir = "Q:\finance-analyzer\scripts\win"
scripts/win\install-crypto-loop-task.ps1:18:
scripts/win\install-crypto-loop-task.ps1:19:$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
scripts/win\install-crypto-loop-task.ps1:20:if ($existing) {
scripts/win\install-crypto-loop-task.ps1:21:    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
scripts/win\install-crypto-loop-task.ps1:22:    Write-Host "Removed existing $TaskName"
scripts/win\install-crypto-loop-task.ps1:23:}
scripts/win\install-crypto-loop-task.ps1:24:
scripts/win\install-crypto-loop-task.ps1:25:$action = New-ScheduledTaskAction -Execute "cmd.exe" `
scripts/win\install-crypto-loop-task.ps1:26:    -Argument "/c `"$scriptDir\crypto-loop.bat`"" `
scripts/win\install-crypto-loop-task.ps1:27:    -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-crypto-loop-task.ps1:28:
scripts/win\install-crypto-loop-task.ps1:29:$trigger1 = New-ScheduledTaskTrigger -AtLogOn
scripts/win\install-crypto-loop-task.ps1:30:$trigger2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday,Saturday,Sunday -At "07:00"
scripts/win\install-crypto-loop-task.ps1:31:
scripts/win\install-crypto-loop-task.ps1:32:# Crypto trades 24/7 - Sat+Sun included in the weekly trigger (vs metals
scripts/win\install-crypto-loop-task.ps1:33:# weekday-only). Singleton lock prevents double-start when AtLogOn fires
scripts/win\install-crypto-loop-task.ps1:34:# while a Saturday-morning trigger also runs.
scripts/win\install-crypto-loop-task.ps1:35:$settings = New-ScheduledTaskSettingsSet `
scripts/win\install-crypto-loop-task.ps1:36:    -AllowStartIfOnBatteries `
scripts/win\install-crypto-loop-task.ps1:37:    -DontStopIfGoingOnBatteries `
scripts/win\install-crypto-loop-task.ps1:38:    -StartWhenAvailable `
scripts/win\install-crypto-loop-task.ps1:39:    -MultipleInstances IgnoreNew `
scripts/win\install-crypto-loop-task.ps1:40:    -ExecutionTimeLimit (New-TimeSpan -Days 3) `
scripts/win\install-crypto-loop-task.ps1:41:    -RestartCount 3 `
scripts/win\install-crypto-loop-task.ps1:42:    -RestartInterval (New-TimeSpan -Minutes 1)
scripts/win\install-crypto-loop-task.ps1:43:
scripts/win\install-crypto-loop-task.ps1:44:Register-ScheduledTask -TaskName $TaskName `
scripts/win\install-crypto-loop-task.ps1:45:    -Action $action -Trigger $trigger1,$trigger2 -Settings $settings `
scripts/win\install-crypto-loop-task.ps1:46:    -Description "Crypto BTC+ETH paper-mode swing loop. Runs scripts\win\crypto-loop.bat. DRY_RUN=True until manually flipped via data\crypto_swing_config.py."
scripts/win\install-crypto-loop-task.ps1:47:
scripts/win\install-crypto-loop-task.ps1:48:Write-Host "Registered $TaskName (NOT started - DRY_RUN=True)"
scripts/win\install-crypto-loop-task.ps1:49:Write-Host "To start now: Start-ScheduledTask -TaskName '$TaskName'"
scripts/win\install-crypto-loop-task.ps1:50:Write-Host "To stop:      Stop-ScheduledTask -TaskName '$TaskName'"
scripts/win\install-crypto-loop-task.ps1:51:Write-Host "Logs:         data\crypto_loop_out.txt"
scripts/win\install-crypto-loop-task.ps1:52:Write-Host "Heartbeat:    data\crypto_loop.heartbeat"
scripts/win\ft-test.bat:1:@echo off
scripts/win\ft-test.bat:2:REM Run pytest for unit and integration tests.
scripts/win\ft-test.bat:3:REM Usage: ft-test.bat [pytest args...]
scripts/win\ft-test.bat:4:REM Example: ft-test.bat tests/unit/ -v
scripts/win\ft-test.bat:5:
scripts/win\ft-test.bat:6:setlocal
scripts/win\ft-test.bat:7:set "PROJECT_DIR=%~dp0..\.."
scripts/win\ft-test.bat:8:call "%PROJECT_DIR%\.venv\Scripts\activate.bat"
scripts/win\ft-test.bat:9:
scripts/win\ft-test.bat:10:if "%1"=="" (
scripts/win\ft-test.bat:11:    python -m pytest "%PROJECT_DIR%\tests" -v
scripts/win\ft-test.bat:12:) else (
scripts/win\ft-test.bat:13:    python -m pytest %*
scripts/win\ft-test.bat:14:)
scripts/win\golddigger-loop.bat:1:@echo off
scripts/win\golddigger-loop.bat:2:title GoldDigger Signal Tracker (auto-restart)
scripts/win\golddigger-loop.bat:3:cd /d Q:\finance-analyzer
scripts/win\golddigger-loop.bat:4:
scripts/win\golddigger-loop.bat:5::restart
scripts/win\golddigger-loop.bat:6:echo [%date% %time%] Starting GoldDigger...
scripts/win\golddigger-loop.bat:7:.venv\Scripts\python.exe -u -m portfolio.golddigger --dry-run >> data\golddigger_out.txt 2>&1
scripts/win\golddigger-loop.bat:8:set EXIT_CODE=%ERRORLEVEL%
scripts/win\golddigger-loop.bat:9:echo [%date% %time%] GoldDigger exited (code %EXIT_CODE%).
scripts/win\golddigger-loop.bat:10:
scripts/win\golddigger-loop.bat:11:REM Check if within market hours (07:00-22:00 CET)
scripts/win\golddigger-loop.bat:12:for /f "tokens=1-2 delims=:" %%a in ("%time: =0%") do set HOUR=%%a
scripts/win\golddigger-loop.bat:13:if %HOUR% GEQ 22 (
scripts/win\golddigger-loop.bat:14:    echo [%date% %time%] Outside market hours -- stopping.
scripts/win\golddigger-loop.bat:15:    goto :eof
scripts/win\golddigger-loop.bat:16:)
scripts/win\golddigger-loop.bat:17:if %HOUR% LSS 7 (
scripts/win\golddigger-loop.bat:18:    echo [%date% %time%] Outside market hours -- stopping.
scripts/win\golddigger-loop.bat:19:    goto :eof
scripts/win\golddigger-loop.bat:20:)
scripts/win\golddigger-loop.bat:21:
scripts/win\golddigger-loop.bat:22:echo [%date% %time%] Restarting in 30s...
scripts/win\golddigger-loop.bat:23:timeout /t 30 /nobreak >nul
scripts/win\golddigger-loop.bat:24:goto restart
scripts/win\golddigger.bat:1:@echo off
scripts/win\golddigger.bat:2:title GoldDigger Signal Tracker
scripts/win\golddigger.bat:3:cd /d Q:\finance-analyzer
scripts/win\golddigger.bat:4:
scripts/win\golddigger.bat:5::restart
scripts/win\golddigger.bat:6:echo [%date% %time%] Starting GoldDigger...
scripts/win\golddigger.bat:7:.venv\Scripts\python.exe -u -m portfolio.golddigger %* >> data\golddigger_out.txt 2>&1
scripts/win\golddigger.bat:8:echo [%date% %time%] GoldDigger exited (code %ERRORLEVEL%). Restarting in 30s...
scripts/win\golddigger.bat:9:timeout /t 30 /nobreak >nul
scripts/win\golddigger.bat:10:goto restart
scripts/win\install-health-check-tasks.ps1:1:# Install PF-HealthCheck scheduled tasks (3 tiers)
scripts/win\install-health-check-tasks.ps1:2:# Run as Administrator
scripts/win\install-health-check-tasks.ps1:3:
scripts/win\install-health-check-tasks.ps1:4:$pythonExe = "Q:\finance-analyzer\.venv\Scripts\python.exe"
scripts/win\install-health-check-tasks.ps1:5:$script = "Q:\finance-analyzer\scripts\health_check.py"
scripts/win\install-health-check-tasks.ps1:6:$workDir = "Q:\finance-analyzer"
scripts/win\install-health-check-tasks.ps1:7:
scripts/win\install-health-check-tasks.ps1:8:# Tier 1: Full check at 11:00 CET (09:00 UTC summer / 10:00 UTC winter)
scripts/win\install-health-check-tasks.ps1:9:$action1 = New-ScheduledTaskAction -Execute $pythonExe -Argument "-u $script --tier full" -WorkingDirectory $workDir
scripts/win\install-health-check-tasks.ps1:10:$trigger1 = New-ScheduledTaskTrigger -Daily -At "11:00"
scripts/win\install-health-check-tasks.ps1:11:$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
scripts/win\install-health-check-tasks.ps1:12:Register-ScheduledTask -TaskName "PF-HealthCheck-Full" -Action $action1 -Trigger $trigger1 -Settings $settings -Description "System health contract - full check (11:00 CET)" -Force
scripts/win\install-health-check-tasks.ps1:13:
scripts/win\install-health-check-tasks.ps1:14:# Tier 2: Pre-US check at 15:25 CET (13:25 UTC summer / 14:25 UTC winter)
scripts/win\install-health-check-tasks.ps1:15:$action2 = New-ScheduledTaskAction -Execute $pythonExe -Argument "-u $script --tier pre-us" -WorkingDirectory $workDir
scripts/win\install-health-check-tasks.ps1:16:$trigger2 = New-ScheduledTaskTrigger -Daily -At "15:25"
scripts/win\install-health-check-tasks.ps1:17:Register-ScheduledTask -TaskName "PF-HealthCheck-PreUS" -Action $action2 -Trigger $trigger2 -Settings $settings -Description "System health contract - pre-US-open check (15:25 CET)" -Force
scripts/win\install-health-check-tasks.ps1:18:
scripts/win\install-health-check-tasks.ps1:19:# Tier 3: Post-US check at 22:05 CET (20:05 UTC summer / 21:05 UTC winter)
scripts/win\install-health-check-tasks.ps1:20:$action3 = New-ScheduledTaskAction -Execute $pythonExe -Argument "-u $script --tier post-us" -WorkingDirectory $workDir
scripts/win\install-health-check-tasks.ps1:21:$trigger3 = New-ScheduledTaskTrigger -Daily -At "22:05"
scripts/win\install-health-check-tasks.ps1:22:Register-ScheduledTask -TaskName "PF-HealthCheck-PostUS" -Action $action3 -Trigger $trigger3 -Settings $settings -Description "System health contract - post-US-close check (22:05 CET)" -Force
scripts/win\install-health-check-tasks.ps1:23:
scripts/win\install-health-check-tasks.ps1:24:Write-Host "Installed 3 health check tasks:"
scripts/win\install-health-check-tasks.ps1:25:Get-ScheduledTask | Where-Object {$_.TaskName -like 'PF-HealthCheck*'} | Select-Object TaskName, State | Format-Table -AutoSize
scripts/win\install-adversarial-review-task.ps1:1:# Install PF-AdversarialReview scheduled task
scripts/win\install-adversarial-review-task.ps1:2:# Run as: powershell -ExecutionPolicy Bypass -File scripts\win\install-adversarial-review-task.ps1
scripts/win\install-adversarial-review-task.ps1:3:#
scripts/win\install-adversarial-review-task.ps1:4:# Schedule:
scripts/win\install-adversarial-review-task.ps1:5:#   Daily at 17:20 CET/CEST (after market close, before after-hours research)
scripts/win\install-adversarial-review-task.ps1:6:#   Runs Claude Code CLI with /fgl protocol for dual adversarial review
scripts/win\install-adversarial-review-task.ps1:7:
scripts/win\install-adversarial-review-task.ps1:8:$taskName = "PF-AdversarialReview"
scripts/win\install-adversarial-review-task.ps1:9:$scriptPath = "Q:\finance-analyzer\scripts\win\adversarial-review.bat"
scripts/win\install-adversarial-review-task.ps1:10:
scripts/win\install-adversarial-review-task.ps1:11:# Remove existing task if present
scripts/win\install-adversarial-review-task.ps1:12:$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
scripts/win\install-adversarial-review-task.ps1:13:if ($existing) {
scripts/win\install-adversarial-review-task.ps1:14:    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
scripts/win\install-adversarial-review-task.ps1:15:    Write-Host "Removed existing task: $taskName"
scripts/win\install-adversarial-review-task.ps1:16:}
scripts/win\install-adversarial-review-task.ps1:17:
scripts/win\install-adversarial-review-task.ps1:18:$action = New-ScheduledTaskAction `
scripts/win\install-adversarial-review-task.ps1:19:    -Execute $scriptPath `
scripts/win\install-adversarial-review-task.ps1:20:    -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-adversarial-review-task.ps1:21:
scripts/win\install-adversarial-review-task.ps1:22:$trigger = New-ScheduledTaskTrigger -Daily -At "17:20"
scripts/win\install-adversarial-review-task.ps1:23:
scripts/win\install-adversarial-review-task.ps1:24:$settings = New-ScheduledTaskSettingsSet `
scripts/win\install-adversarial-review-task.ps1:25:    -AllowStartIfOnBatteries `
scripts/win\install-adversarial-review-task.ps1:26:    -DontStopIfGoingOnBatteries `
scripts/win\install-adversarial-review-task.ps1:27:    -StartWhenAvailable `
scripts/win\install-adversarial-review-task.ps1:28:    -ExecutionTimeLimit (New-TimeSpan -Hours 2)
scripts/win\install-adversarial-review-task.ps1:29:
scripts/win\install-adversarial-review-task.ps1:30:Register-ScheduledTask -TaskName $taskName `
scripts/win\install-adversarial-review-task.ps1:31:    -Action $action -Trigger $trigger -Settings $settings `
scripts/win\install-adversarial-review-task.ps1:32:    -Description "Daily dual adversarial review of finance-analyzer codebase (Codex + Claude with cross-critique). Uses /fgl protocol."
scripts/win\install-adversarial-review-task.ps1:33:
scripts/win\install-adversarial-review-task.ps1:34:Write-Host ""
scripts/win\install-adversarial-review-task.ps1:35:Write-Host "Installed: $taskName"
scripts/win\install-adversarial-review-task.ps1:36:Write-Host "Schedule:  Daily at 17:20 local time"
scripts/win\install-adversarial-review-task.ps1:37:Write-Host "Script:    $scriptPath"
scripts/win\install-adversarial-review-task.ps1:38:Write-Host "Output:    Q:\finance-analyzer\data\adversarial_review_out.txt"
scripts/win\install-adversarial-review-task.ps1:39:Write-Host "Timeout:   2 hours max"
scripts/win\install-adversarial-review-task.ps1:40:Write-Host ""
scripts/win\install-adversarial-review-task.ps1:41:Write-Host "To run manually: Start-ScheduledTask -TaskName '$taskName'"
scripts/win\install-adversarial-review-task.ps1:42:Write-Host "To check status: Get-ScheduledTask -TaskName '$taskName'"
scripts/win\install-adversarial-review-task.ps1:43:Write-Host "To remove:       Unregister-ScheduledTask -TaskName '$taskName'"
scripts/win\install-golddigger-task.ps1:1:param(
scripts/win\install-golddigger-task.ps1:2:    [string]$TaskName = "PF-GoldDigger",
scripts/win\install-golddigger-task.ps1:3:    [switch]$Remove
scripts/win\install-golddigger-task.ps1:4:)
scripts/win\install-golddigger-task.ps1:5:
scripts/win\install-golddigger-task.ps1:6:$ErrorActionPreference = "Stop"
scripts/win\install-golddigger-task.ps1:7:
scripts/win\install-golddigger-task.ps1:8:$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
scripts/win\install-golddigger-task.ps1:9:$runner = Join-Path $repoRoot "scripts\win\golddigger.bat"
scripts/win\install-golddigger-task.ps1:10:
scripts/win\install-golddigger-task.ps1:11:if (-not (Test-Path $runner)) {
scripts/win\install-golddigger-task.ps1:12:    throw "Runner not found: $runner"
scripts/win\install-golddigger-task.ps1:13:}
scripts/win\install-golddigger-task.ps1:14:
scripts/win\install-golddigger-task.ps1:15:if ($Remove) {
scripts/win\install-golddigger-task.ps1:16:    & schtasks /Delete /TN $TaskName /F
scripts/win\install-golddigger-task.ps1:17:    exit $LASTEXITCODE
scripts/win\install-golddigger-task.ps1:18:}
scripts/win\install-golddigger-task.ps1:19:
scripts/win\install-golddigger-task.ps1:20:$taskCommand = '"' + $runner + '" --live'
scripts/win\install-golddigger-task.ps1:21:
scripts/win\install-golddigger-task.ps1:22:& schtasks /Create /TN $TaskName /SC ONLOGON /TR $taskCommand /F
scripts/win\install-golddigger-task.ps1:23:if ($LASTEXITCODE -ne 0) {
scripts/win\install-golddigger-task.ps1:24:    exit $LASTEXITCODE
scripts/win\install-golddigger-task.ps1:25:}
scripts/win\install-golddigger-task.ps1:26:
scripts/win\install-golddigger-task.ps1:27:Write-Host "Created or updated task: $TaskName"
scripts/win\install-golddigger-task.ps1:28:Write-Host "Schedule: on logon"
scripts/win\install-golddigger-task.ps1:29:Write-Host "Command: $taskCommand"
scripts/win\install-log-rotate-task.ps1:1:# Install the scheduled task that runs portfolio/log_rotation.py hourly.
scripts/win\install-log-rotate-task.ps1:2:# Run as:
scripts/win\install-log-rotate-task.ps1:3:#   powershell -ExecutionPolicy Bypass -File scripts\win\install-log-rotate-task.ps1
scripts/win\install-log-rotate-task.ps1:4:#
scripts/win\install-log-rotate-task.ps1:5:# Why hourly: loop_out.txt grows ~600KB/h at full verbosity. With a 5MB
scripts/win\install-log-rotate-task.ps1:6:# rotate threshold that's a ~8h cycle, so an hourly check keeps total
scripts/win\install-log-rotate-task.ps1:7:# (live + .1 .. .5.gz) under ~30MB. JSONL files in the policy mostly
scripts/win\install-log-rotate-task.ps1:8:# use age-based archive — checking them hourly is harmless idempotent.
scripts/win\install-log-rotate-task.ps1:9:#
scripts/win\install-log-rotate-task.ps1:10:# This script DOES auto-start the task once registered. Rotation is
scripts/win\install-log-rotate-task.ps1:11:# read-only-ish (rename + truncate) — no risk to live trading.
scripts/win\install-log-rotate-task.ps1:12:
scripts/win\install-log-rotate-task.ps1:13:$TaskName = "PF-LogRotate"
scripts/win\install-log-rotate-task.ps1:14:$pythonExe = "Q:\finance-analyzer\.venv\Scripts\python.exe"
scripts/win\install-log-rotate-task.ps1:15:$workDir   = "Q:\finance-analyzer"
scripts/win\install-log-rotate-task.ps1:16:
scripts/win\install-log-rotate-task.ps1:17:$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
scripts/win\install-log-rotate-task.ps1:18:if ($existing) {
scripts/win\install-log-rotate-task.ps1:19:    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
scripts/win\install-log-rotate-task.ps1:20:    Write-Host "Removed existing $TaskName"
scripts/win\install-log-rotate-task.ps1:21:}
scripts/win\install-log-rotate-task.ps1:22:
scripts/win\install-log-rotate-task.ps1:23:$action = New-ScheduledTaskAction -Execute $pythonExe `
scripts/win\install-log-rotate-task.ps1:24:    -Argument "-m portfolio.log_rotation" `
scripts/win\install-log-rotate-task.ps1:25:    -WorkingDirectory $workDir
scripts/win\install-log-rotate-task.ps1:26:
scripts/win\install-log-rotate-task.ps1:27:# Hourly trigger starting in 5 minutes (so first run isn't immediately
scripts/win\install-log-rotate-task.ps1:28:# on install).
scripts/win\install-log-rotate-task.ps1:29:$startBoundary = (Get-Date).AddMinutes(5)
scripts/win\install-log-rotate-task.ps1:30:$trigger = New-ScheduledTaskTrigger -Once -At $startBoundary `
scripts/win\install-log-rotate-task.ps1:31:    -RepetitionInterval (New-TimeSpan -Hours 1)
scripts/win\install-log-rotate-task.ps1:32:
scripts/win\install-log-rotate-task.ps1:33:$settings = New-ScheduledTaskSettingsSet `
scripts/win\install-log-rotate-task.ps1:34:    -AllowStartIfOnBatteries `
scripts/win\install-log-rotate-task.ps1:35:    -DontStopIfGoingOnBatteries `
scripts/win\install-log-rotate-task.ps1:36:    -StartWhenAvailable `
scripts/win\install-log-rotate-task.ps1:37:    -MultipleInstances IgnoreNew `
scripts/win\install-log-rotate-task.ps1:38:    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
scripts/win\install-log-rotate-task.ps1:39:    -RestartCount 1 `
scripts/win\install-log-rotate-task.ps1:40:    -RestartInterval (New-TimeSpan -Minutes 1)
scripts/win\install-log-rotate-task.ps1:41:
scripts/win\install-log-rotate-task.ps1:42:Register-ScheduledTask -TaskName $TaskName `
scripts/win\install-log-rotate-task.ps1:43:    -Action $action -Trigger $trigger -Settings $settings `
scripts/win\install-log-rotate-task.ps1:44:    -Description "Hourly log rotation for finance-analyzer. Runs portfolio.log_rotation. Rotates loop_out.txt, golddigger_out.txt (size-based) and JSONL files (age-based). Archive: data\archive\."
scripts/win\install-log-rotate-task.ps1:45:
scripts/win\install-log-rotate-task.ps1:46:Write-Host "Registered $TaskName (next run: $startBoundary)"
scripts/win\install-log-rotate-task.ps1:47:Write-Host "To run now:      Start-ScheduledTask -TaskName '$TaskName'"
scripts/win\install-log-rotate-task.ps1:48:Write-Host "To inspect dry:  $pythonExe -m portfolio.log_rotation --dry-run"
scripts/win\install-log-rotate-task.ps1:49:Write-Host "To see sizes:    $pythonExe -m portfolio.log_rotation --status"
scripts/win\install-log-rotate-task.ps1:50:Write-Host "Archive dir:     data\archive\"
scripts/win\install-loop-health-report-task.ps1:1:# One-time scheduled task: paper-mode health check 2 weeks after the
scripts/win\install-loop-health-report-task.ps1:2:# 2026-05-01 midfinance merge.
scripts/win\install-loop-health-report-task.ps1:3:#
scripts/win\install-loop-health-report-task.ps1:4:# Run as:
scripts/win\install-loop-health-report-task.ps1:5:#   powershell -ExecutionPolicy Bypass -File scripts\win\install-loop-health-report-task.ps1
scripts/win\install-loop-health-report-task.ps1:6:#
scripts/win\install-loop-health-report-task.ps1:7:# Fires once on 2026-05-15 at 18:00 local time (post-EU close, before
scripts/win\install-loop-health-report-task.ps1:8:# US close). The task auto-disables itself after firing. To re-arm for
scripts/win\install-loop-health-report-task.ps1:9:# a later date, edit $RunOnce below and re-run this script.
scripts/win\install-loop-health-report-task.ps1:10:
scripts/win\install-loop-health-report-task.ps1:11:$TaskName = "PF-LoopHealthReport-20260515"
scripts/win\install-loop-health-report-task.ps1:12:$RunOnce = "2026-05-15T18:00:00"   # local time (Europe/Stockholm)
scripts/win\install-loop-health-report-task.ps1:13:
scripts/win\install-loop-health-report-task.ps1:14:$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
scripts/win\install-loop-health-report-task.ps1:15:if ($existing) {
scripts/win\install-loop-health-report-task.ps1:16:    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
scripts/win\install-loop-health-report-task.ps1:17:    Write-Host "Removed existing $TaskName"
scripts/win\install-loop-health-report-task.ps1:18:}
scripts/win\install-loop-health-report-task.ps1:19:
scripts/win\install-loop-health-report-task.ps1:20:$action = New-ScheduledTaskAction -Execute "Q:\finance-analyzer\.venv\Scripts\python.exe" `
scripts/win\install-loop-health-report-task.ps1:21:    -Argument "Q:\finance-analyzer\scripts\loop_health_report.py" `
scripts/win\install-loop-health-report-task.ps1:22:    -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-loop-health-report-task.ps1:23:
scripts/win\install-loop-health-report-task.ps1:24:$trigger = New-ScheduledTaskTrigger -Once -At $RunOnce
scripts/win\install-loop-health-report-task.ps1:25:
scripts/win\install-loop-health-report-task.ps1:26:$settings = New-ScheduledTaskSettingsSet `
scripts/win\install-loop-health-report-task.ps1:27:    -AllowStartIfOnBatteries `
scripts/win\install-loop-health-report-task.ps1:28:    -DontStopIfGoingOnBatteries `
scripts/win\install-loop-health-report-task.ps1:29:    -StartWhenAvailable `
scripts/win\install-loop-health-report-task.ps1:30:    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
scripts/win\install-loop-health-report-task.ps1:31:
scripts/win\install-loop-health-report-task.ps1:32:Register-ScheduledTask -TaskName $TaskName `
scripts/win\install-loop-health-report-task.ps1:33:    -Action $action -Trigger $trigger -Settings $settings `
scripts/win\install-loop-health-report-task.ps1:34:    -Description "One-time paper-mode health check for crypto+MSTR+oil swing loops. Fires 2026-05-15 18:00 local. Sends Telegram summary."
scripts/win\install-loop-health-report-task.ps1:35:
scripts/win\install-loop-health-report-task.ps1:36:Write-Host "Registered $TaskName for $RunOnce"
scripts/win\install-loop-health-report-task.ps1:37:Write-Host "To verify:    Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
scripts/win\install-loop-health-report-task.ps1:38:Write-Host "To run early: Start-ScheduledTask -TaskName '$TaskName'"
scripts/win\install-loop-health-report-task.ps1:39:Write-Host "To cancel:    Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
scripts/win\install-loop-health-watchdog-task.ps1:1:# Install the periodic loop-health watchdog scheduled task.
scripts/win\install-loop-health-watchdog-task.ps1:2:#
scripts/win\install-loop-health-watchdog-task.ps1:3:# Run as:
scripts/win\install-loop-health-watchdog-task.ps1:4:#   powershell -ExecutionPolicy Bypass -File scripts\win\install-loop-health-watchdog-task.ps1
scripts/win\install-loop-health-watchdog-task.ps1:5:#
scripts/win\install-loop-health-watchdog-task.ps1:6:# Fires every 30 minutes from logon onward. Sends a consolidated telegram
scripts/win\install-loop-health-watchdog-task.ps1:7:# alert when any loop heartbeat is stale or missing. Per-loop cooldown
scripts/win\install-loop-health-watchdog-task.ps1:8:# (4h default in scripts/loop_health_watchdog.py) prevents alert spam
scripts/win\install-loop-health-watchdog-task.ps1:9:# from a persistently-dead loop.
scripts/win\install-loop-health-watchdog-task.ps1:10:
scripts/win\install-loop-health-watchdog-task.ps1:11:$TaskName = "PF-LoopHealthWatchdog"
scripts/win\install-loop-health-watchdog-task.ps1:12:$scriptDir = "Q:\finance-analyzer\scripts"
scripts/win\install-loop-health-watchdog-task.ps1:13:
scripts/win\install-loop-health-watchdog-task.ps1:14:$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
scripts/win\install-loop-health-watchdog-task.ps1:15:if ($existing) {
scripts/win\install-loop-health-watchdog-task.ps1:16:    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
scripts/win\install-loop-health-watchdog-task.ps1:17:    Write-Host "Removed existing $TaskName"
scripts/win\install-loop-health-watchdog-task.ps1:18:}
scripts/win\install-loop-health-watchdog-task.ps1:19:
scripts/win\install-loop-health-watchdog-task.ps1:20:$action = New-ScheduledTaskAction -Execute "Q:\finance-analyzer\.venv\Scripts\python.exe" `
scripts/win\install-loop-health-watchdog-task.ps1:21:    -Argument "$scriptDir\loop_health_watchdog.py" `
scripts/win\install-loop-health-watchdog-task.ps1:22:    -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-loop-health-watchdog-task.ps1:23:
scripts/win\install-loop-health-watchdog-task.ps1:24:# Trigger every 30 minutes, indefinitely, starting at logon
scripts/win\install-loop-health-watchdog-task.ps1:25:$trigger = New-ScheduledTaskTrigger -AtLogOn
scripts/win\install-loop-health-watchdog-task.ps1:26:$trigger.Repetition = (New-ScheduledTaskTrigger -Once -At (Get-Date) `
scripts/win\install-loop-health-watchdog-task.ps1:27:    -RepetitionInterval (New-TimeSpan -Minutes 30) `
scripts/win\install-loop-health-watchdog-task.ps1:28:    -RepetitionDuration (New-TimeSpan -Days 365)).Repetition
scripts/win\install-loop-health-watchdog-task.ps1:29:
scripts/win\install-loop-health-watchdog-task.ps1:30:$settings = New-ScheduledTaskSettingsSet `
scripts/win\install-loop-health-watchdog-task.ps1:31:    -AllowStartIfOnBatteries `
scripts/win\install-loop-health-watchdog-task.ps1:32:    -DontStopIfGoingOnBatteries `
scripts/win\install-loop-health-watchdog-task.ps1:33:    -StartWhenAvailable `
scripts/win\install-loop-health-watchdog-task.ps1:34:    -MultipleInstances IgnoreNew `
scripts/win\install-loop-health-watchdog-task.ps1:35:    -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
scripts/win\install-loop-health-watchdog-task.ps1:36:
scripts/win\install-loop-health-watchdog-task.ps1:37:Register-ScheduledTask -TaskName $TaskName `
scripts/win\install-loop-health-watchdog-task.ps1:38:    -Action $action -Trigger $trigger -Settings $settings `
scripts/win\install-loop-health-watchdog-task.ps1:39:    -Description "Periodic loop-health watchdog. Reads data/*_loop.heartbeat every 30min and sends telegram alerts on stale/missing. 4h cooldown per loop."
scripts/win\install-loop-health-watchdog-task.ps1:40:
scripts/win\install-loop-health-watchdog-task.ps1:41:Write-Host "Registered $TaskName (every 30min)"
scripts/win\install-loop-health-watchdog-task.ps1:42:Write-Host "To verify:    Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
scripts/win\install-loop-health-watchdog-task.ps1:43:Write-Host "To run now:   Start-ScheduledTask -TaskName '$TaskName'"
scripts/win\install-loop-health-watchdog-task.ps1:44:Write-Host "To stop:      Stop-ScheduledTask -TaskName '$TaskName'"
scripts/win\install-loop-health-watchdog-task.ps1:45:Write-Host "Cooldown state: data\loop_health_watchdog_state.json"
scripts/win\install-loop-resume-task.ps1:1:# install-loop-resume-task.ps1 — Run as Administrator
scripts/win\install-loop-resume-task.ps1:2:# Creates PF-LoopResume: fires on wake-from-sleep, runs pf-loop-ensure.ps1
scripts/win\install-loop-resume-task.ps1:3:
scripts/win\install-loop-resume-task.ps1:4:$taskName = "PF-LoopResume"
scripts/win\install-loop-resume-task.ps1:5:$scriptPath = "Q:\finance-analyzer\scripts\win\pf-loop-ensure.ps1"
scripts/win\install-loop-resume-task.ps1:6:
scripts/win\install-loop-resume-task.ps1:7:# Remove existing task if present
scripts/win\install-loop-resume-task.ps1:8:Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
scripts/win\install-loop-resume-task.ps1:9:
scripts/win\install-loop-resume-task.ps1:10:# Trigger: Event ID 1 from Microsoft-Windows-Power-Troubleshooter = resume from sleep
scripts/win\install-loop-resume-task.ps1:11:$trigger = New-ScheduledTaskTrigger -AtLogOn  # placeholder, replaced by XML below
scripts/win\install-loop-resume-task.ps1:12:
scripts/win\install-loop-resume-task.ps1:13:# Action: run the ensure script
scripts/win\install-loop-resume-task.ps1:14:$action = New-ScheduledTaskAction `
scripts/win\install-loop-resume-task.ps1:15:    -Execute "powershell.exe" `
scripts/win\install-loop-resume-task.ps1:16:    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
scripts/win\install-loop-resume-task.ps1:17:
scripts/win\install-loop-resume-task.ps1:18:# Settings
scripts/win\install-loop-resume-task.ps1:19:$settings = New-ScheduledTaskSettingsSet `
scripts/win\install-loop-resume-task.ps1:20:    -AllowStartIfOnBatteries `
scripts/win\install-loop-resume-task.ps1:21:    -DontStopIfGoingOnBatteries `
scripts/win\install-loop-resume-task.ps1:22:    -StartWhenAvailable `
scripts/win\install-loop-resume-task.ps1:23:    -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
scripts/win\install-loop-resume-task.ps1:24:
scripts/win\install-loop-resume-task.ps1:25:# Register with logon trigger first (we'll add the event trigger via XML)
scripts/win\install-loop-resume-task.ps1:26:$task = Register-ScheduledTask `
scripts/win\install-loop-resume-task.ps1:27:    -TaskName $taskName `
scripts/win\install-loop-resume-task.ps1:28:    -Action $action `
scripts/win\install-loop-resume-task.ps1:29:    -Trigger $trigger `
scripts/win\install-loop-resume-task.ps1:30:    -Settings $settings `
scripts/win\install-loop-resume-task.ps1:31:    -Description "Ensures pf-loop.bat is running after wake-from-sleep or logon" `
scripts/win\install-loop-resume-task.ps1:32:    -RunLevel Highest
scripts/win\install-loop-resume-task.ps1:33:
scripts/win\install-loop-resume-task.ps1:34:# Now add the Event trigger (Power-Troubleshooter Event ID 1 = resume from suspend)
scripts/win\install-loop-resume-task.ps1:35:# Export, modify XML, re-import
scripts/win\install-loop-resume-task.ps1:36:$xml = Export-ScheduledTask -TaskName $taskName
scripts/win\install-loop-resume-task.ps1:37:
scripts/win\install-loop-resume-task.ps1:38:# Insert event trigger XML before </Triggers>
scripts/win\install-loop-resume-task.ps1:39:$eventTrigger = @"
scripts/win\install-loop-resume-task.ps1:40:    <EventTrigger>
scripts/win\install-loop-resume-task.ps1:41:      <Enabled>true</Enabled>
scripts/win\install-loop-resume-task.ps1:42:      <Subscription>&lt;QueryList&gt;&lt;Query Id="0" Path="System"&gt;&lt;Select Path="System"&gt;*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]&lt;/Select&gt;&lt;/Query&gt;&lt;/QueryList&gt;</Subscription>
scripts/win\install-loop-resume-task.ps1:43:    </EventTrigger>
scripts/win\install-loop-resume-task.ps1:44:"@
scripts/win\install-loop-resume-task.ps1:45:
scripts/win\install-loop-resume-task.ps1:46:$xml = $xml -replace '</Triggers>', "$eventTrigger`n  </Triggers>"
scripts/win\install-loop-resume-task.ps1:47:
scripts/win\install-loop-resume-task.ps1:48:# Re-register with updated XML
scripts/win\install-loop-resume-task.ps1:49:Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
scripts/win\install-loop-resume-task.ps1:50:Register-ScheduledTask -TaskName $taskName -Xml $xml
scripts/win\install-loop-resume-task.ps1:51:
scripts/win\install-loop-resume-task.ps1:52:Write-Host ""
scripts/win\install-loop-resume-task.ps1:53:Write-Host "=== $taskName installed ==="
scripts/win\install-loop-resume-task.ps1:54:Write-Host "Triggers: (1) At logon, (2) On wake from sleep (Event ID 1)"
scripts/win\install-loop-resume-task.ps1:55:Write-Host "Action: powershell -File $scriptPath"
scripts/win\install-loop-resume-task.ps1:56:Write-Host ""
scripts/win\install-loop-resume-task.ps1:57:Write-Host "To verify: schtasks /Query /TN '$taskName' /V /FO LIST"
scripts/win\install-loop-health-daily-task.ps1:1:# Install the DAILY loop-health summary scheduled task.
scripts/win\install-loop-health-daily-task.ps1:2:#
scripts/win\install-loop-health-daily-task.ps1:3:# Run as:
scripts/win\install-loop-health-daily-task.ps1:4:#   powershell -ExecutionPolicy Bypass -File scripts\win\install-loop-health-daily-task.ps1
scripts/win\install-loop-health-daily-task.ps1:5:#
scripts/win\install-loop-health-daily-task.ps1:6:# Fires every morning at 08:00 local time. Sends a Telegram summary with:
scripts/win\install-loop-health-daily-task.ps1:7:#   - heartbeat freshness for crypto + oil
scripts/win\install-loop-health-daily-task.ps1:8:#   - mstr_loop_poll.jsonl existence (mstr proxy)
scripts/win\install-loop-health-daily-task.ps1:9:#   - oil + mstr scorecard rollups (paper-mode trade counts, win rate,
scripts/win\install-loop-health-daily-task.ps1:10:#     time-to-live-flip readiness gates)
scripts/win\install-loop-health-daily-task.ps1:11:#
scripts/win\install-loop-health-daily-task.ps1:12:# Complements two other scheduled tasks:
scripts/win\install-loop-health-daily-task.ps1:13:#   - PF-LoopHealthWatchdog (every 30min) — alerts on stale/missing
scripts/win\install-loop-health-daily-task.ps1:14:#     heartbeats with 4h cooldown.
scripts/win\install-loop-health-daily-task.ps1:15:#   - PF-LoopHealthReport-20260515 (one-shot, T+2w) — auto-disables.
scripts/win\install-loop-health-daily-task.ps1:16:#
scripts/win\install-loop-health-daily-task.ps1:17:# This daily task is the "watchdog of the watchdog" — if the daily
scripts/win\install-loop-health-daily-task.ps1:18:# summary stops arriving, you know the WHOLE monitoring chain is dead
scripts/win\install-loop-health-daily-task.ps1:19:# (Telegram, scheduled-task service, the script itself), not just one
scripts/win\install-loop-health-daily-task.ps1:20:# loop. Cheap insurance: one Telegram per day.
scripts/win\install-loop-health-daily-task.ps1:21:
scripts/win\install-loop-health-daily-task.ps1:22:$TaskName = "PF-LoopHealthDaily"
scripts/win\install-loop-health-daily-task.ps1:23:$scriptDir = "Q:\finance-analyzer\scripts"
scripts/win\install-loop-health-daily-task.ps1:24:
scripts/win\install-loop-health-daily-task.ps1:25:$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
scripts/win\install-loop-health-daily-task.ps1:26:if ($existing) {
scripts/win\install-loop-health-daily-task.ps1:27:    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
scripts/win\install-loop-health-daily-task.ps1:28:    Write-Host "Removed existing $TaskName"
scripts/win\install-loop-health-daily-task.ps1:29:}
scripts/win\install-loop-health-daily-task.ps1:30:
scripts/win\install-loop-health-daily-task.ps1:31:$action = New-ScheduledTaskAction -Execute "Q:\finance-analyzer\.venv\Scripts\python.exe" `
scripts/win\install-loop-health-daily-task.ps1:32:    -Argument "$scriptDir\loop_health_report.py" `
scripts/win\install-loop-health-daily-task.ps1:33:    -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-loop-health-daily-task.ps1:34:
scripts/win\install-loop-health-daily-task.ps1:35:# 08:00 local — well after EU pre-open prep (07:00) so the loops have
scripts/win\install-loop-health-daily-task.ps1:36:# logged a few cycles, and before the user's typical morning check.
scripts/win\install-loop-health-daily-task.ps1:37:$trigger = New-ScheduledTaskTrigger -Daily -At "08:00"
scripts/win\install-loop-health-daily-task.ps1:38:
scripts/win\install-loop-health-daily-task.ps1:39:$settings = New-ScheduledTaskSettingsSet `
scripts/win\install-loop-health-daily-task.ps1:40:    -AllowStartIfOnBatteries `
scripts/win\install-loop-health-daily-task.ps1:41:    -DontStopIfGoingOnBatteries `
scripts/win\install-loop-health-daily-task.ps1:42:    -StartWhenAvailable `
scripts/win\install-loop-health-daily-task.ps1:43:    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
scripts/win\install-loop-health-daily-task.ps1:44:
scripts/win\install-loop-health-daily-task.ps1:45:Register-ScheduledTask -TaskName $TaskName `
scripts/win\install-loop-health-daily-task.ps1:46:    -Action $action -Trigger $trigger -Settings $settings `
scripts/win\install-loop-health-daily-task.ps1:47:    -Description "Daily loop-health summary at 08:00 local. Sends Telegram with heartbeat freshness + paper-mode scorecards. Insurance against the WHOLE monitoring chain dying silently."
scripts/win\install-loop-health-daily-task.ps1:48:
scripts/win\install-loop-health-daily-task.ps1:49:Write-Host "Registered $TaskName (daily 08:00 local)"
scripts/win\install-loop-health-daily-task.ps1:50:Write-Host "To verify:    Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
scripts/win\install-loop-health-daily-task.ps1:51:Write-Host "To run now:   Start-ScheduledTask -TaskName '$TaskName'"
scripts/win\install-loop-health-daily-task.ps1:52:Write-Host "To stop:      Stop-ScheduledTask -TaskName '$TaskName'"
scripts/win\install-meta-learner-task.ps1:1:# Install PF-MetaLearnerRetrain scheduled task
scripts/win\install-meta-learner-task.ps1:2:# Run as: powershell -ExecutionPolicy Bypass -File scripts\win\install-meta-learner-task.ps1
scripts/win\install-meta-learner-task.ps1:3:#
scripts/win\install-meta-learner-task.ps1:4:# Schedule:
scripts/win\install-meta-learner-task.ps1:5:#   Daily at 19:00 CET (after market close, after PF-OutcomeCheck at 18:00)
scripts/win\install-meta-learner-task.ps1:6:#   Low priority (start /LOW + os.nice(19) + num_threads=1) to avoid disrupting trading loop
scripts/win\install-meta-learner-task.ps1:7:
scripts/win\install-meta-learner-task.ps1:8:$taskName = "PF-MetaLearnerRetrain"
scripts/win\install-meta-learner-task.ps1:9:$scriptPath = "Q:\finance-analyzer\scripts\win\meta-learner-retrain.bat"
scripts/win\install-meta-learner-task.ps1:10:
scripts/win\install-meta-learner-task.ps1:11:# Remove existing task if present
scripts/win\install-meta-learner-task.ps1:12:$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
scripts/win\install-meta-learner-task.ps1:13:if ($existing) {
scripts/win\install-meta-learner-task.ps1:14:    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
scripts/win\install-meta-learner-task.ps1:15:    Write-Host "Removed existing task: $taskName"
scripts/win\install-meta-learner-task.ps1:16:}
scripts/win\install-meta-learner-task.ps1:17:
scripts/win\install-meta-learner-task.ps1:18:$action = New-ScheduledTaskAction `
scripts/win\install-meta-learner-task.ps1:19:    -Execute $scriptPath `
scripts/win\install-meta-learner-task.ps1:20:    -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-meta-learner-task.ps1:21:
scripts/win\install-meta-learner-task.ps1:22:$trigger = New-ScheduledTaskTrigger -Daily -At "19:00"
scripts/win\install-meta-learner-task.ps1:23:
scripts/win\install-meta-learner-task.ps1:24:$settings = New-ScheduledTaskSettingsSet `
scripts/win\install-meta-learner-task.ps1:25:    -AllowStartIfOnBatteries `
scripts/win\install-meta-learner-task.ps1:26:    -DontStopIfGoingOnBatteries `
scripts/win\install-meta-learner-task.ps1:27:    -StartWhenAvailable `
scripts/win\install-meta-learner-task.ps1:28:    -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
scripts/win\install-meta-learner-task.ps1:29:
scripts/win\install-meta-learner-task.ps1:30:Register-ScheduledTask -TaskName $taskName `
scripts/win\install-meta-learner-task.ps1:31:    -Action $action -Trigger $trigger -Settings $settings `
scripts/win\install-meta-learner-task.ps1:32:    -Description "Daily LightGBM meta-learner retraining (low priority, 1 thread)"
scripts/win\install-meta-learner-task.ps1:33:
scripts/win\install-meta-learner-task.ps1:34:Write-Host ""
scripts/win\install-meta-learner-task.ps1:35:Write-Host "Installed: $taskName"
scripts/win\install-meta-learner-task.ps1:36:Write-Host "Schedule:  Daily at 19:00 (after PF-OutcomeCheck at 18:00)"
scripts/win\install-meta-learner-task.ps1:37:Write-Host "Script:    $scriptPath"
scripts/win\install-meta-learner-task.ps1:38:Write-Host "Output:    Q:\finance-analyzer\data\meta_learner_retrain_out.txt"
scripts/win\install-meta-learner-task.ps1:39:Write-Host ""
scripts/win\install-meta-learner-task.ps1:40:Write-Host "To run manually: Start-ScheduledTask -TaskName '$taskName'"
scripts/win\install-meta-learner-task.ps1:41:Write-Host "To check status: Get-ScheduledTask -TaskName '$taskName'"
scripts/win\install-meta-learner-task.ps1:42:Write-Host "To remove:       Unregister-ScheduledTask -TaskName '$taskName'"
scripts/win\install-market-tasks.ps1:1:# Install scheduled tasks for Silver Monitor and GoldDigger
scripts/win\install-market-tasks.ps1:2:# Run as: powershell -ExecutionPolicy Bypass -File scripts\win\install-market-tasks.ps1
scripts/win\install-market-tasks.ps1:3:#
scripts/win\install-market-tasks.ps1:4:# Schedule:
scripts/win\install-market-tasks.ps1:5:#   Start: 07:00 CET daily (Mon-Fri) = EU market pre-open
scripts/win\install-market-tasks.ps1:6:#   Auto-stop: bat files exit after 22:00 CET
scripts/win\install-market-tasks.ps1:7:#   Auto-restart on crash: bat loop with 30s delay
scripts/win\install-market-tasks.ps1:8:
scripts/win\install-market-tasks.ps1:9:$scriptDir = "Q:\finance-analyzer\scripts\win"
scripts/win\install-market-tasks.ps1:10:
scripts/win\install-market-tasks.ps1:11:# --- PF-MetalsLoop ---
scripts/win\install-market-tasks.ps1:12:& "$scriptDir\install-metals-loop-task.ps1"
scripts/win\install-market-tasks.ps1:13:Write-Host ""
scripts/win\install-market-tasks.ps1:14:
scripts/win\install-market-tasks.ps1:15:# --- PF-SilverMonitor ---
scripts/win\install-market-tasks.ps1:16:$taskName1 = "PF-SilverMonitor"
scripts/win\install-market-tasks.ps1:17:$existing1 = Get-ScheduledTask -TaskName $taskName1 -ErrorAction SilentlyContinue
scripts/win\install-market-tasks.ps1:18:if ($existing1) {
scripts/win\install-market-tasks.ps1:19:    Unregister-ScheduledTask -TaskName $taskName1 -Confirm:$false
scripts/win\install-market-tasks.ps1:20:    Write-Host "Removed existing $taskName1"
scripts/win\install-market-tasks.ps1:21:}
scripts/win\install-market-tasks.ps1:22:
scripts/win\install-market-tasks.ps1:23:$action1 = New-ScheduledTaskAction -Execute "cmd.exe" `
scripts/win\install-market-tasks.ps1:24:    -Argument "/c `"$scriptDir\silver-monitor.bat`"" `
scripts/win\install-market-tasks.ps1:25:    -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-market-tasks.ps1:26:
scripts/win\install-market-tasks.ps1:27:$trigger1 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "07:00"
scripts/win\install-market-tasks.ps1:28:$trigger1.Repetition = $null
scripts/win\install-market-tasks.ps1:29:$triggerLogon1 = New-ScheduledTaskTrigger -AtLogon
scripts/win\install-market-tasks.ps1:30:
scripts/win\install-market-tasks.ps1:31:$settings1 = New-ScheduledTaskSettingsSet `
scripts/win\install-market-tasks.ps1:32:    -AllowStartIfOnBatteries `
scripts/win\install-market-tasks.ps1:33:    -DontStopIfGoingOnBatteries `
scripts/win\install-market-tasks.ps1:34:    -StartWhenAvailable `
scripts/win\install-market-tasks.ps1:35:    -ExecutionTimeLimit (New-TimeSpan -Hours 16) `
scripts/win\install-market-tasks.ps1:36:    -RestartCount 3 `
scripts/win\install-market-tasks.ps1:37:    -RestartInterval (New-TimeSpan -Minutes 1)
scripts/win\install-market-tasks.ps1:38:
scripts/win\install-market-tasks.ps1:39:Register-ScheduledTask -TaskName $taskName1 `
scripts/win\install-market-tasks.ps1:40:    -Action $action1 -Trigger @($trigger1, $triggerLogon1) -Settings $settings1 `
scripts/win\install-market-tasks.ps1:41:    -Description "Silver price monitor with Claude analysis. Starts at logon and 07:00 CET weekdays. Auto-restarts on crash."
scripts/win\install-market-tasks.ps1:42:
scripts/win\install-market-tasks.ps1:43:Write-Host "Registered $taskName1 (logon + 07:00 CET Mon-Fri)"
scripts/win\install-market-tasks.ps1:44:
scripts/win\install-market-tasks.ps1:45:# --- PF-GoldDigger ---
scripts/win\install-market-tasks.ps1:46:$taskName2 = "PF-GoldDigger"
scripts/win\install-market-tasks.ps1:47:$existing2 = Get-ScheduledTask -TaskName $taskName2 -ErrorAction SilentlyContinue
scripts/win\install-market-tasks.ps1:48:if ($existing2) {
scripts/win\install-market-tasks.ps1:49:    Unregister-ScheduledTask -TaskName $taskName2 -Confirm:$false
scripts/win\install-market-tasks.ps1:50:    Write-Host "Removed existing $taskName2"
scripts/win\install-market-tasks.ps1:51:}
scripts/win\install-market-tasks.ps1:52:
scripts/win\install-market-tasks.ps1:53:$action2 = New-ScheduledTaskAction -Execute "cmd.exe" `
scripts/win\install-market-tasks.ps1:54:    -Argument "/c `"$scriptDir\golddigger-loop.bat`"" `
scripts/win\install-market-tasks.ps1:55:    -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-market-tasks.ps1:56:
scripts/win\install-market-tasks.ps1:57:$trigger2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "07:00"
scripts/win\install-market-tasks.ps1:58:
scripts/win\install-market-tasks.ps1:59:$settings2 = New-ScheduledTaskSettingsSet `
scripts/win\install-market-tasks.ps1:60:    -AllowStartIfOnBatteries `
scripts/win\install-market-tasks.ps1:61:    -DontStopIfGoingOnBatteries `
scripts/win\install-market-tasks.ps1:62:    -StartWhenAvailable `
scripts/win\install-market-tasks.ps1:63:    -ExecutionTimeLimit (New-TimeSpan -Hours 16) `
scripts/win\install-market-tasks.ps1:64:    -RestartCount 3 `
scripts/win\install-market-tasks.ps1:65:    -RestartInterval (New-TimeSpan -Minutes 1)
scripts/win\install-market-tasks.ps1:66:
scripts/win\install-market-tasks.ps1:67:Register-ScheduledTask -TaskName $taskName2 `
scripts/win\install-market-tasks.ps1:68:    -Action $action2 -Trigger $trigger2 -Settings $settings2 `
scripts/win\install-market-tasks.ps1:69:    -Description "GoldDigger intraday gold signal tracker (dry-run). Auto-restarts on crash. Runs 07:00-22:00 CET Mon-Fri."
scripts/win\install-market-tasks.ps1:70:
scripts/win\install-market-tasks.ps1:71:Write-Host "Registered $taskName2 (07:00 CET Mon-Fri)"
scripts/win\install-market-tasks.ps1:72:
scripts/win\install-market-tasks.ps1:73:# --- Summary ---
scripts/win\install-market-tasks.ps1:74:Write-Host ""
scripts/win\install-market-tasks.ps1:75:Write-Host "=== Scheduled Tasks Installed ==="
scripts/win\install-market-tasks.ps1:76:Write-Host "PF-MetalsLoop:   on logon + 07:00 CET Mon-Fri, auto-restart on crash"
scripts/win\install-market-tasks.ps1:77:Write-Host "PF-SilverMonitor: on logon + 07:00 CET Mon-Fri, auto-restart on crash"
scripts/win\install-market-tasks.ps1:78:Write-Host "PF-GoldDigger:    07:00-22:00 CET Mon-Fri, auto-restart on crash"
scripts/win\install-market-tasks.ps1:79:Write-Host ""
scripts/win\install-market-tasks.ps1:80:Write-Host "To start NOW:  Start-ScheduledTask -TaskName 'PF-MetalsLoop'"
scripts/win\install-market-tasks.ps1:81:Write-Host "               Start-ScheduledTask -TaskName 'PF-SilverMonitor'"
scripts/win\install-market-tasks.ps1:82:Write-Host "               Start-ScheduledTask -TaskName 'PF-GoldDigger'"
scripts/win\install-market-tasks.ps1:83:Write-Host "To stop:       Stop-ScheduledTask -TaskName 'PF-SilverMonitor'"
scripts/win\install-market-tasks.ps1:84:Write-Host "To remove:     Unregister-ScheduledTask -TaskName 'PF-SilverMonitor'"
scripts/win\install-metals-loop-task.ps1:1:# Install the canonical scheduled task for the brokered metals loop.
scripts/win\install-metals-loop-task.ps1:2:# Run as:
scripts/win\install-metals-loop-task.ps1:3:#   powershell -ExecutionPolicy Bypass -File scripts\win\install-metals-loop-task.ps1
scripts/win\install-metals-loop-task.ps1:4:
scripts/win\install-metals-loop-task.ps1:5:$TaskName = "PF-MetalsLoop"
scripts/win\install-metals-loop-task.ps1:6:$scriptDir = "Q:\finance-analyzer\scripts\win"
scripts/win\install-metals-loop-task.ps1:7:
scripts/win\install-metals-loop-task.ps1:8:$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
scripts/win\install-metals-loop-task.ps1:9:if ($existing) {
scripts/win\install-metals-loop-task.ps1:10:    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
scripts/win\install-metals-loop-task.ps1:11:    Write-Host "Removed existing $TaskName"
scripts/win\install-metals-loop-task.ps1:12:}
scripts/win\install-metals-loop-task.ps1:13:
scripts/win\install-metals-loop-task.ps1:14:$action = New-ScheduledTaskAction -Execute "cmd.exe" `
scripts/win\install-metals-loop-task.ps1:15:    -Argument "/c `"$scriptDir\metals-loop.bat`"" `
scripts/win\install-metals-loop-task.ps1:16:    -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-metals-loop-task.ps1:17:
scripts/win\install-metals-loop-task.ps1:18:$trigger1 = New-ScheduledTaskTrigger -AtLogOn
scripts/win\install-metals-loop-task.ps1:19:$trigger2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "07:00"
scripts/win\install-metals-loop-task.ps1:20:
scripts/win\install-metals-loop-task.ps1:21:$settings = New-ScheduledTaskSettingsSet `
scripts/win\install-metals-loop-task.ps1:22:    -AllowStartIfOnBatteries `
scripts/win\install-metals-loop-task.ps1:23:    -DontStopIfGoingOnBatteries `
scripts/win\install-metals-loop-task.ps1:24:    -StartWhenAvailable `
scripts/win\install-metals-loop-task.ps1:25:    -MultipleInstances IgnoreNew `
scripts/win\install-metals-loop-task.ps1:26:    -ExecutionTimeLimit (New-TimeSpan -Days 3) `
scripts/win\install-metals-loop-task.ps1:27:    -RestartCount 3 `
scripts/win\install-metals-loop-task.ps1:28:    -RestartInterval (New-TimeSpan -Minutes 1)
scripts/win\install-metals-loop-task.ps1:29:
scripts/win\install-metals-loop-task.ps1:30:Register-ScheduledTask -TaskName $TaskName `
scripts/win\install-metals-loop-task.ps1:31:    -Action $action -Trigger $trigger1,$trigger2 -Settings $settings `
scripts/win\install-metals-loop-task.ps1:32:    -Description "Brokered metals execution loop. Runs scripts\\win\\metals-loop.bat and auto-restarts on crash."
scripts/win\install-metals-loop-task.ps1:33:
scripts/win\install-metals-loop-task.ps1:34:Write-Host "Registered $TaskName"
scripts/win\install-metals-loop-task.ps1:35:Write-Host "To start now: Start-ScheduledTask -TaskName '$TaskName'"
scripts/win\install-metals-loop-task.ps1:36:Write-Host "To stop:      Stop-ScheduledTask -TaskName '$TaskName'"
scripts/win\install-mstr-loop-task.ps1:1:# Install the canonical scheduled task for the MSTR loop.
scripts/win\install-mstr-loop-task.ps1:2:# Run as:
scripts/win\install-mstr-loop-task.ps1:3:#   powershell -ExecutionPolicy Bypass -File scripts\win\install-mstr-loop-task.ps1
scripts/win\install-mstr-loop-task.ps1:4:#
scripts/win\install-mstr-loop-task.ps1:5:# Defaults to PHASE=shadow via MSTR_LOOP_PHASE env var (per
scripts/win\install-mstr-loop-task.ps1:6:# docs/MSTR_LOOP_NOTES.md). Phase A (live) requires 90 days of shadow data
scripts/win\install-mstr-loop-task.ps1:7:# and explicit user approval — flip MSTR_LOOP_PHASE=live in the task's
scripts/win\install-mstr-loop-task.ps1:8:# action only after that gate clears.
scripts/win\install-mstr-loop-task.ps1:9:
scripts/win\install-mstr-loop-task.ps1:10:$TaskName = "PF-MstrLoop"
scripts/win\install-mstr-loop-task.ps1:11:$scriptDir = "Q:\finance-analyzer\scripts\win"
scripts/win\install-mstr-loop-task.ps1:12:
scripts/win\install-mstr-loop-task.ps1:13:$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
scripts/win\install-mstr-loop-task.ps1:14:if ($existing) {
scripts/win\install-mstr-loop-task.ps1:15:    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
scripts/win\install-mstr-loop-task.ps1:16:    Write-Host "Removed existing $TaskName"
scripts/win\install-mstr-loop-task.ps1:17:}
scripts/win\install-mstr-loop-task.ps1:18:
scripts/win\install-mstr-loop-task.ps1:19:# We use cmd /c with `set` to inject the phase env var before launching the
scripts/win\install-mstr-loop-task.ps1:20:# wrapper. Going via cmd avoids the PowerShell-vs-batch quoting hell.
scripts/win\install-mstr-loop-task.ps1:21:$action = New-ScheduledTaskAction -Execute "cmd.exe" `
scripts/win\install-mstr-loop-task.ps1:22:    -Argument "/c `"set `"MSTR_LOOP_PHASE=shadow`"&&`"$scriptDir\mstr-loop.bat`"`"" `
scripts/win\install-mstr-loop-task.ps1:23:    -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-mstr-loop-task.ps1:24:
scripts/win\install-mstr-loop-task.ps1:25:$trigger1 = New-ScheduledTaskTrigger -AtLogOn
scripts/win\install-mstr-loop-task.ps1:26:# MSTR is US-listed — only relevant during US session (15:30-22:00 CET).
scripts/win\install-mstr-loop-task.ps1:27:# The loop itself early-exits outside session, but starting earlier means
scripts/win\install-mstr-loop-task.ps1:28:# the heartbeat is fresh by the time the open hits.
scripts/win\install-mstr-loop-task.ps1:29:$trigger2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "14:00"
scripts/win\install-mstr-loop-task.ps1:30:
scripts/win\install-mstr-loop-task.ps1:31:$settings = New-ScheduledTaskSettingsSet `
scripts/win\install-mstr-loop-task.ps1:32:    -AllowStartIfOnBatteries `
scripts/win\install-mstr-loop-task.ps1:33:    -DontStopIfGoingOnBatteries `
scripts/win\install-mstr-loop-task.ps1:34:    -StartWhenAvailable `
scripts/win\install-mstr-loop-task.ps1:35:    -MultipleInstances IgnoreNew `
scripts/win\install-mstr-loop-task.ps1:36:    -ExecutionTimeLimit (New-TimeSpan -Days 1) `
scripts/win\install-mstr-loop-task.ps1:37:    -RestartCount 3 `
scripts/win\install-mstr-loop-task.ps1:38:    -RestartInterval (New-TimeSpan -Minutes 1)
scripts/win\install-mstr-loop-task.ps1:39:
scripts/win\install-mstr-loop-task.ps1:40:Register-ScheduledTask -TaskName $TaskName `
scripts/win\install-mstr-loop-task.ps1:41:    -Action $action -Trigger $trigger1,$trigger2 -Settings $settings `
scripts/win\install-mstr-loop-task.ps1:42:    -Description "MSTR shadow-mode loop. Runs scripts\win\mstr-loop.bat with PHASE=shadow. Decisions logged to data\mstr_loop_shadow.jsonl, no live orders. Phase A requires 90d shadow data + manual approval."
scripts/win\install-mstr-loop-task.ps1:43:
scripts/win\install-mstr-loop-task.ps1:44:Write-Host "Registered $TaskName (PHASE=shadow)"
scripts/win\install-mstr-loop-task.ps1:45:Write-Host "To start now: Start-ScheduledTask -TaskName '$TaskName'"
scripts/win\install-mstr-loop-task.ps1:46:Write-Host "To stop:      Stop-ScheduledTask -TaskName '$TaskName'"
scripts/win\install-mstr-loop-task.ps1:47:Write-Host "Logs:         logs\mstr_loop_out.txt"
scripts/win\install-mstr-loop-task.ps1:48:Write-Host "Shadow log:   data\mstr_loop_shadow.jsonl"
scripts/win\install-mstr-loop-task.ps1:49:Write-Host "Phase notes:  docs\MSTR_LOOP_NOTES.md"
scripts/win\install-oil-loop-task.ps1:1:# Install the canonical scheduled task for the oil (WTI) loop.
scripts/win\install-oil-loop-task.ps1:2:# Run as:
scripts/win\install-oil-loop-task.ps1:3:#   powershell -ExecutionPolicy Bypass -File scripts\win\install-oil-loop-task.ps1
scripts/win\install-oil-loop-task.ps1:4:#
scripts/win\install-oil-loop-task.ps1:5:# Parity with PF-CryptoLoop / PF-MetalsLoop. Oil futures trade nearly 24/7
scripts/win\install-oil-loop-task.ps1:6:# on CME (Sun 23:00 CET to Fri 22:00 CET), so the weekly trigger covers
scripts/win\install-oil-loop-task.ps1:7:# Mon-Fri (Saturday is the only fully closed day).
scripts/win\install-oil-loop-task.ps1:8:
scripts/win\install-oil-loop-task.ps1:9:$TaskName = "PF-OilLoop"
scripts/win\install-oil-loop-task.ps1:10:$scriptDir = "Q:\finance-analyzer\scripts\win"
scripts/win\install-oil-loop-task.ps1:11:
scripts/win\install-oil-loop-task.ps1:12:$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
scripts/win\install-oil-loop-task.ps1:13:if ($existing) {
scripts/win\install-oil-loop-task.ps1:14:    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
scripts/win\install-oil-loop-task.ps1:15:    Write-Host "Removed existing $TaskName"
scripts/win\install-oil-loop-task.ps1:16:}
scripts/win\install-oil-loop-task.ps1:17:
scripts/win\install-oil-loop-task.ps1:18:$action = New-ScheduledTaskAction -Execute "cmd.exe" `
scripts/win\install-oil-loop-task.ps1:19:    -Argument "/c `"$scriptDir\oil-loop.bat`"" `
scripts/win\install-oil-loop-task.ps1:20:    -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-oil-loop-task.ps1:21:
scripts/win\install-oil-loop-task.ps1:22:$trigger1 = New-ScheduledTaskTrigger -AtLogOn
scripts/win\install-oil-loop-task.ps1:23:$trigger2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday,Monday,Tuesday,Wednesday,Thursday,Friday -At "07:00"
scripts/win\install-oil-loop-task.ps1:24:
scripts/win\install-oil-loop-task.ps1:25:$settings = New-ScheduledTaskSettingsSet `
scripts/win\install-oil-loop-task.ps1:26:    -AllowStartIfOnBatteries `
scripts/win\install-oil-loop-task.ps1:27:    -DontStopIfGoingOnBatteries `
scripts/win\install-oil-loop-task.ps1:28:    -StartWhenAvailable `
scripts/win\install-oil-loop-task.ps1:29:    -MultipleInstances IgnoreNew `
scripts/win\install-oil-loop-task.ps1:30:    -ExecutionTimeLimit (New-TimeSpan -Days 3) `
scripts/win\install-oil-loop-task.ps1:31:    -RestartCount 3 `
scripts/win\install-oil-loop-task.ps1:32:    -RestartInterval (New-TimeSpan -Minutes 1)
scripts/win\install-oil-loop-task.ps1:33:
scripts/win\install-oil-loop-task.ps1:34:Register-ScheduledTask -TaskName $TaskName `
scripts/win\install-oil-loop-task.ps1:35:    -Action $action -Trigger $trigger1,$trigger2 -Settings $settings `
scripts/win\install-oil-loop-task.ps1:36:    -Description "Oil WTI paper-mode swing loop. Runs scripts\win\oil-loop.bat. DRY_RUN=True until manually flipped via data\oil_swing_config.py."
scripts/win\install-oil-loop-task.ps1:37:
scripts/win\install-oil-loop-task.ps1:38:Write-Host "Registered $TaskName (NOT started - DRY_RUN=True)"
scripts/win\install-oil-loop-task.ps1:39:Write-Host "To start now: Start-ScheduledTask -TaskName '$TaskName'"
scripts/win\install-oil-loop-task.ps1:40:Write-Host "To stop:      Stop-ScheduledTask -TaskName '$TaskName'"
scripts/win\install-oil-loop-task.ps1:41:Write-Host "Logs:         data\oil_loop_out.txt"
scripts/win\install-oil-loop-task.ps1:42:Write-Host "Heartbeat:    data\oil_loop.heartbeat"
scripts/win\install-oil-loop-task.ps1:43:Write-Host ""
scripts/win\install-oil-loop-task.ps1:44:Write-Host "First-run prerequisite: oil warrant catalog must be populated."
scripts/win\install-oil-loop-task.ps1:45:Write-Host "Run a one-shot probe with a live Avanza session to fill the catalog:"
scripts/win\install-oil-loop-task.ps1:46:Write-Host "  .venv\Scripts\python.exe -u data\oil_loop.py --once --debug"
scripts/win\install-rc-keepalive-task.ps1:1:# install-rc-keepalive-task.ps1 — Run as Administrator
scripts/win\install-rc-keepalive-task.ps1:2:# Creates a scheduled task that runs rc-keepalive.ps1 every 5 minutes.
scripts/win\install-rc-keepalive-task.ps1:3:#
scripts/win\install-rc-keepalive-task.ps1:4:# Anthropic's server-side TTL is ~20 min without real user activity.
scripts/win\install-rc-keepalive-task.ps1:5:# Keepalive recycles idle servers at staggered thresholds (13/15/17 min)
scripts/win\install-rc-keepalive-task.ps1:6:# to keep them visible in the claude.ai/code picker.
scripts/win\install-rc-keepalive-task.ps1:7:#
scripts/win\install-rc-keepalive-task.ps1:8:# Also creates a wake-from-sleep trigger that runs keepalive in -Wake mode,
scripts/win\install-rc-keepalive-task.ps1:9:# which immediately recycles all idle servers (sleep guarantees staleness).
scripts/win\install-rc-keepalive-task.ps1:10:
scripts/win\install-rc-keepalive-task.ps1:11:$taskName   = "PF-RCKeepalive"
scripts/win\install-rc-keepalive-task.ps1:12:$scriptPath = "Q:\finance-analyzer\scripts\win\rc-keepalive.ps1"
scripts/win\install-rc-keepalive-task.ps1:13:
scripts/win\install-rc-keepalive-task.ps1:14:# Remove existing task if present
scripts/win\install-rc-keepalive-task.ps1:15:Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
scripts/win\install-rc-keepalive-task.ps1:16:
scripts/win\install-rc-keepalive-task.ps1:17:$actionPeriodic = New-ScheduledTaskAction `
scripts/win\install-rc-keepalive-task.ps1:18:    -Execute "powershell.exe" `
scripts/win\install-rc-keepalive-task.ps1:19:    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
scripts/win\install-rc-keepalive-task.ps1:20:
scripts/win\install-rc-keepalive-task.ps1:21:$actionWake = New-ScheduledTaskAction `
scripts/win\install-rc-keepalive-task.ps1:22:    -Execute "powershell.exe" `
scripts/win\install-rc-keepalive-task.ps1:23:    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" -Wake"
scripts/win\install-rc-keepalive-task.ps1:24:
scripts/win\install-rc-keepalive-task.ps1:25:# Trigger 1: every 5 minutes, starting now, repeating for ~25 years (max safe duration)
scripts/win\install-rc-keepalive-task.ps1:26:$triggerPeriodic = New-ScheduledTaskTrigger -Once -At (Get-Date) `
scripts/win\install-rc-keepalive-task.ps1:27:    -RepetitionInterval (New-TimeSpan -Minutes 5) `
scripts/win\install-rc-keepalive-task.ps1:28:    -RepetitionDuration (New-TimeSpan -Days 9000)
scripts/win\install-rc-keepalive-task.ps1:29:
scripts/win\install-rc-keepalive-task.ps1:30:$settings = New-ScheduledTaskSettingsSet `
scripts/win\install-rc-keepalive-task.ps1:31:    -AllowStartIfOnBatteries `
scripts/win\install-rc-keepalive-task.ps1:32:    -DontStopIfGoingOnBatteries `
scripts/win\install-rc-keepalive-task.ps1:33:    -StartWhenAvailable `
scripts/win\install-rc-keepalive-task.ps1:34:    -ExecutionTimeLimit (New-TimeSpan -Minutes 2) `
scripts/win\install-rc-keepalive-task.ps1:35:    -RestartCount 3 `
scripts/win\install-rc-keepalive-task.ps1:36:    -RestartInterval (New-TimeSpan -Minutes 1)
scripts/win\install-rc-keepalive-task.ps1:37:
scripts/win\install-rc-keepalive-task.ps1:38:# Register with periodic trigger first
scripts/win\install-rc-keepalive-task.ps1:39:Register-ScheduledTask `
scripts/win\install-rc-keepalive-task.ps1:40:    -TaskName $taskName `
scripts/win\install-rc-keepalive-task.ps1:41:    -Action $actionPeriodic `
scripts/win\install-rc-keepalive-task.ps1:42:    -Trigger $triggerPeriodic `
scripts/win\install-rc-keepalive-task.ps1:43:    -Settings $settings `
scripts/win\install-rc-keepalive-task.ps1:44:    -Description "RC server keepalive: recycle idle servers before 20-min Anthropic TTL (every 5 min, staggered 13/15/17 min thresholds)" `
scripts/win\install-rc-keepalive-task.ps1:45:    -RunLevel Highest
scripts/win\install-rc-keepalive-task.ps1:46:
scripts/win\install-rc-keepalive-task.ps1:47:# Add wake-from-sleep trigger via XML modification
scripts/win\install-rc-keepalive-task.ps1:48:# PowerShell's New-ScheduledTaskTrigger doesn't support event triggers, so we
scripts/win\install-rc-keepalive-task.ps1:49:# export the task XML, inject the wake trigger, and re-register.
scripts/win\install-rc-keepalive-task.ps1:50:try {
scripts/win\install-rc-keepalive-task.ps1:51:    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
scripts/win\install-rc-keepalive-task.ps1:52:    $xml = [xml](Export-ScheduledTask -TaskName $taskName)
scripts/win\install-rc-keepalive-task.ps1:53:    $ns = $xml.Task.NamespaceURI
scripts/win\install-rc-keepalive-task.ps1:54:
scripts/win\install-rc-keepalive-task.ps1:55:    # Create EventTrigger node for Power-Troubleshooter EventID 1 (system wake)
scripts/win\install-rc-keepalive-task.ps1:56:    $wakeTriggerXml = @"
scripts/win\install-rc-keepalive-task.ps1:57:<EventTrigger xmlns="$ns">
scripts/win\install-rc-keepalive-task.ps1:58:  <Enabled>true</Enabled>
scripts/win\install-rc-keepalive-task.ps1:59:  <Delay>PT10S</Delay>
scripts/win\install-rc-keepalive-task.ps1:60:  <Subscription>&lt;QueryList&gt;&lt;Query Id="0" Path="System"&gt;&lt;Select Path="System"&gt;*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]&lt;/Select&gt;&lt;/Query&gt;&lt;/QueryList&gt;</Subscription>
scripts/win\install-rc-keepalive-task.ps1:61:</EventTrigger>
scripts/win\install-rc-keepalive-task.ps1:62:"@
scripts/win\install-rc-keepalive-task.ps1:63:    $frag = $xml.CreateDocumentFragment()
scripts/win\install-rc-keepalive-task.ps1:64:    $frag.InnerXml = $wakeTriggerXml
scripts/win\install-rc-keepalive-task.ps1:65:    $xml.Task.Triggers.AppendChild($frag) | Out-Null
scripts/win\install-rc-keepalive-task.ps1:66:
scripts/win\install-rc-keepalive-task.ps1:67:    # Override the action to include -Wake flag for the event trigger
scripts/win\install-rc-keepalive-task.ps1:68:    # (Can't have per-trigger actions in Task Scheduler, so the periodic action stays as-is.
scripts/win\install-rc-keepalive-task.ps1:69:    #  The ensure script handles wake via -WakeDelay, and the 5-min periodic keepalive will
scripts/win\install-rc-keepalive-task.ps1:70:    #  catch stale servers within 5 min of wake anyway.)
scripts/win\install-rc-keepalive-task.ps1:71:
scripts/win\install-rc-keepalive-task.ps1:72:    Register-ScheduledTask -TaskName $taskName -Xml $xml.OuterXml -Force | Out-Null
scripts/win\install-rc-keepalive-task.ps1:73:    Write-Host "  Wake trigger added successfully (Power-Troubleshooter EventID 1, 10s delay)."
scripts/win\install-rc-keepalive-task.ps1:74:} catch {
scripts/win\install-rc-keepalive-task.ps1:75:    Write-Host "  WARNING: Could not add wake trigger: $_"
scripts/win\install-rc-keepalive-task.ps1:76:    Write-Host "  The periodic 5-min check will still catch stale servers within 5 min of wake."
scripts/win\install-rc-keepalive-task.ps1:77:}
scripts/win\install-rc-keepalive-task.ps1:78:
scripts/win\install-rc-keepalive-task.ps1:79:Write-Host ""
scripts/win\install-rc-keepalive-task.ps1:80:Write-Host "=== Installed ==="
scripts/win\install-rc-keepalive-task.ps1:81:Write-Host "  Task:     $taskName"
scripts/win\install-rc-keepalive-task.ps1:82:Write-Host "  Interval: every 5 minutes (periodic) + on wake-from-sleep (-Wake flag)"
scripts/win\install-rc-keepalive-task.ps1:83:Write-Host "  Script:   $scriptPath"
scripts/win\install-rc-keepalive-task.ps1:84:Write-Host "  Thresholds: Trading=13min, Development=15min, Research=17min"
scripts/win\install-rc-keepalive-task.ps1:85:Write-Host "  Anthropic TTL: ~20 min (margin: 3-7 min)"
scripts/win\install-rc-keepalive-task.ps1:86:Write-Host ""
scripts/win\install-rc-keepalive-task.ps1:87:Write-Host "To verify:"
scripts/win\install-rc-keepalive-task.ps1:88:Write-Host "  schtasks /Query /TN '$taskName' /V /FO LIST"
scripts/win\install-rc-keepalive-task.ps1:89:Write-Host ""
scripts/win\install-rc-keepalive-task.ps1:90:Write-Host "To test now:"
scripts/win\install-rc-keepalive-task.ps1:91:Write-Host "  schtasks /Run /TN '$taskName'"
scripts/win\install-rc-keepalive-task.ps1:92:Write-Host "  # or: powershell -File `"$scriptPath`""
scripts/win\install-rc-keepalive-task.ps1:93:Write-Host "  # wake mode: powershell -File `"$scriptPath`" -Wake"
scripts/win\install-rc-keepalive-task.ps1:94:Write-Host ""
scripts/win\install-rc-server-task.ps1:1:# install-rc-server-task.ps1 — Run as Administrator
scripts/win\install-rc-server-task.ps1:2:# Creates two tasks for always-on Claude Code RC servers:
scripts/win\install-rc-server-task.ps1:3:#   PF-RemoteControl       — At logon (immediate launch, no delay)
scripts/win\install-rc-server-task.ps1:4:#   PF-RemoteControl-Wake  — On wake from sleep (30s delay for auto-reconnect)
scripts/win\install-rc-server-task.ps1:5:
scripts/win\install-rc-server-task.ps1:6:$scriptPath = "Q:\finance-analyzer\scripts\win\rc-server-ensure.ps1"
scripts/win\install-rc-server-task.ps1:7:
scripts/win\install-rc-server-task.ps1:8:# ---------- Task 1: Logon trigger (no delay) ----------
scripts/win\install-rc-server-task.ps1:9:$logonTask = "PF-RemoteControl"
scripts/win\install-rc-server-task.ps1:10:Unregister-ScheduledTask -TaskName $logonTask -Confirm:$false -ErrorAction SilentlyContinue
scripts/win\install-rc-server-task.ps1:11:
scripts/win\install-rc-server-task.ps1:12:$logonAction = New-ScheduledTaskAction `
scripts/win\install-rc-server-task.ps1:13:    -Execute "powershell.exe" `
scripts/win\install-rc-server-task.ps1:14:    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
scripts/win\install-rc-server-task.ps1:15:
scripts/win\install-rc-server-task.ps1:16:$logonTrigger = New-ScheduledTaskTrigger -AtLogOn
scripts/win\install-rc-server-task.ps1:17:
scripts/win\install-rc-server-task.ps1:18:$settings = New-ScheduledTaskSettingsSet `
scripts/win\install-rc-server-task.ps1:19:    -AllowStartIfOnBatteries `
scripts/win\install-rc-server-task.ps1:20:    -DontStopIfGoingOnBatteries `
scripts/win\install-rc-server-task.ps1:21:    -StartWhenAvailable `
scripts/win\install-rc-server-task.ps1:22:    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
scripts/win\install-rc-server-task.ps1:23:    -RestartCount 3 `
scripts/win\install-rc-server-task.ps1:24:    -RestartInterval (New-TimeSpan -Minutes 1)
scripts/win\install-rc-server-task.ps1:25:
scripts/win\install-rc-server-task.ps1:26:Register-ScheduledTask `
scripts/win\install-rc-server-task.ps1:27:    -TaskName $logonTask `
scripts/win\install-rc-server-task.ps1:28:    -Action $logonAction `
scripts/win\install-rc-server-task.ps1:29:    -Trigger $logonTrigger `
scripts/win\install-rc-server-task.ps1:30:    -Settings $settings `
scripts/win\install-rc-server-task.ps1:31:    -Description "Launch Claude Code RC servers on logon" `
scripts/win\install-rc-server-task.ps1:32:    -RunLevel Highest
scripts/win\install-rc-server-task.ps1:33:
scripts/win\install-rc-server-task.ps1:34:# ---------- Task 2: Wake-from-sleep trigger (with -WakeDelay) ----------
scripts/win\install-rc-server-task.ps1:35:$wakeTask = "PF-RemoteControl-Wake"
scripts/win\install-rc-server-task.ps1:36:Unregister-ScheduledTask -TaskName $wakeTask -Confirm:$false -ErrorAction SilentlyContinue
scripts/win\install-rc-server-task.ps1:37:
scripts/win\install-rc-server-task.ps1:38:$wakeAction = New-ScheduledTaskAction `
scripts/win\install-rc-server-task.ps1:39:    -Execute "powershell.exe" `
scripts/win\install-rc-server-task.ps1:40:    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" -WakeDelay"
scripts/win\install-rc-server-task.ps1:41:
scripts/win\install-rc-server-task.ps1:42:# Register with a placeholder trigger first, then replace via XML for event trigger
scripts/win\install-rc-server-task.ps1:43:$placeholderTrigger = New-ScheduledTaskTrigger -AtLogOn
scripts/win\install-rc-server-task.ps1:44:Register-ScheduledTask `
scripts/win\install-rc-server-task.ps1:45:    -TaskName $wakeTask `
scripts/win\install-rc-server-task.ps1:46:    -Action $wakeAction `
scripts/win\install-rc-server-task.ps1:47:    -Trigger $placeholderTrigger `
scripts/win\install-rc-server-task.ps1:48:    -Settings $settings `
scripts/win\install-rc-server-task.ps1:49:    -Description "Check and restart Claude Code RC servers after wake from sleep" `
scripts/win\install-rc-server-task.ps1:50:    -RunLevel Highest
scripts/win\install-rc-server-task.ps1:51:
scripts/win\install-rc-server-task.ps1:52:# Replace logon trigger with wake-from-sleep event trigger
scripts/win\install-rc-server-task.ps1:53:$xml = Export-ScheduledTask -TaskName $wakeTask
scripts/win\install-rc-server-task.ps1:54:
scripts/win\install-rc-server-task.ps1:55:# Remove the placeholder LogonTrigger and insert EventTrigger
scripts/win\install-rc-server-task.ps1:56:$xml = $xml -replace '<LogonTrigger>[\s\S]*?</LogonTrigger>', @"
scripts/win\install-rc-server-task.ps1:57:<EventTrigger>
scripts/win\install-rc-server-task.ps1:58:      <Enabled>true</Enabled>
scripts/win\install-rc-server-task.ps1:59:      <Subscription>&lt;QueryList&gt;&lt;Query Id="0" Path="System"&gt;&lt;Select Path="System"&gt;*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]&lt;/Select&gt;&lt;/Query&gt;&lt;/QueryList&gt;</Subscription>
scripts/win\install-rc-server-task.ps1:60:    </EventTrigger>
scripts/win\install-rc-server-task.ps1:61:"@
scripts/win\install-rc-server-task.ps1:62:
scripts/win\install-rc-server-task.ps1:63:Unregister-ScheduledTask -TaskName $wakeTask -Confirm:$false
scripts/win\install-rc-server-task.ps1:64:Register-ScheduledTask -TaskName $wakeTask -Xml $xml
scripts/win\install-rc-server-task.ps1:65:
scripts/win\install-rc-server-task.ps1:66:# ---------- Summary ----------
scripts/win\install-rc-server-task.ps1:67:Write-Host ""
scripts/win\install-rc-server-task.ps1:68:Write-Host "=== Installed ==="
scripts/win\install-rc-server-task.ps1:69:Write-Host "  $logonTask        — At logon (immediate)"
scripts/win\install-rc-server-task.ps1:70:Write-Host "  $wakeTask   — On wake from sleep (30s delay, connection check)"
scripts/win\install-rc-server-task.ps1:71:Write-Host "  Script: $scriptPath"
scripts/win\install-rc-server-task.ps1:72:Write-Host ""
scripts/win\install-rc-server-task.ps1:73:Write-Host "To verify:"
scripts/win\install-rc-server-task.ps1:74:Write-Host "  schtasks /Query /TN '$logonTask' /V /FO LIST"
scripts/win\install-rc-server-task.ps1:75:Write-Host "  schtasks /Query /TN '$wakeTask' /V /FO LIST"
scripts/win\install-rc-server-task.ps1:76:Write-Host ""
scripts/win\install-rc-server-task.ps1:77:Write-Host "To test now:"
scripts/win\install-rc-server-task.ps1:78:Write-Host "  schtasks /Run /TN '$wakeTask'"
scripts/win\install-rc-server-task.ps1:79:Write-Host ""
scripts/win\install-rc-watchdog-task.ps1:1:# install-rc-watchdog-task.ps1 — Run as Administrator
scripts/win\install-rc-watchdog-task.ps1:2:# Creates a scheduled task that runs rc-watchdog.ps1 every 30 minutes.
scripts/win\install-rc-watchdog-task.ps1:3:# The watchdog proactively recycles RC servers before the 24h session timeout
scripts/win\install-rc-watchdog-task.ps1:4:# and detects/kills zombies. Sends Telegram alerts on any action.
scripts/win\install-rc-watchdog-task.ps1:5:
scripts/win\install-rc-watchdog-task.ps1:6:$taskName   = "PF-RC-Watchdog"
scripts/win\install-rc-watchdog-task.ps1:7:$scriptPath = "Q:\finance-analyzer\scripts\win\rc-watchdog.ps1"
scripts/win\install-rc-watchdog-task.ps1:8:
scripts/win\install-rc-watchdog-task.ps1:9:# Remove existing task if present
scripts/win\install-rc-watchdog-task.ps1:10:Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
scripts/win\install-rc-watchdog-task.ps1:11:
scripts/win\install-rc-watchdog-task.ps1:12:$action = New-ScheduledTaskAction `
scripts/win\install-rc-watchdog-task.ps1:13:    -Execute "powershell.exe" `
scripts/win\install-rc-watchdog-task.ps1:14:    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
scripts/win\install-rc-watchdog-task.ps1:15:
scripts/win\install-rc-watchdog-task.ps1:16:# Trigger: every 30 minutes, starting now, repeating indefinitely
scripts/win\install-rc-watchdog-task.ps1:17:$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `
scripts/win\install-rc-watchdog-task.ps1:18:    -RepetitionInterval (New-TimeSpan -Minutes 30) `
scripts/win\install-rc-watchdog-task.ps1:19:    -RepetitionDuration ([TimeSpan]::MaxValue)
scripts/win\install-rc-watchdog-task.ps1:20:
scripts/win\install-rc-watchdog-task.ps1:21:$settings = New-ScheduledTaskSettingsSet `
scripts/win\install-rc-watchdog-task.ps1:22:    -AllowStartIfOnBatteries `
scripts/win\install-rc-watchdog-task.ps1:23:    -DontStopIfGoingOnBatteries `
scripts/win\install-rc-watchdog-task.ps1:24:    -StartWhenAvailable `
scripts/win\install-rc-watchdog-task.ps1:25:    -ExecutionTimeLimit (New-TimeSpan -Minutes 2) `
scripts/win\install-rc-watchdog-task.ps1:26:    -RestartCount 3 `
scripts/win\install-rc-watchdog-task.ps1:27:    -RestartInterval (New-TimeSpan -Minutes 1)
scripts/win\install-rc-watchdog-task.ps1:28:
scripts/win\install-rc-watchdog-task.ps1:29:Register-ScheduledTask `
scripts/win\install-rc-watchdog-task.ps1:30:    -TaskName $taskName `
scripts/win\install-rc-watchdog-task.ps1:31:    -Action $action `
scripts/win\install-rc-watchdog-task.ps1:32:    -Trigger $trigger `
scripts/win\install-rc-watchdog-task.ps1:33:    -Settings $settings `
scripts/win\install-rc-watchdog-task.ps1:34:    -Description "RC server watchdog: proactive 20h recycle + zombie detection + Telegram alerts (every 30 min)" `
scripts/win\install-rc-watchdog-task.ps1:35:    -RunLevel Highest
scripts/win\install-rc-watchdog-task.ps1:36:
scripts/win\install-rc-watchdog-task.ps1:37:Write-Host ""
scripts/win\install-rc-watchdog-task.ps1:38:Write-Host "=== Installed ==="
scripts/win\install-rc-watchdog-task.ps1:39:Write-Host "  Task:     $taskName"
scripts/win\install-rc-watchdog-task.ps1:40:Write-Host "  Interval: every 30 minutes"
scripts/win\install-rc-watchdog-task.ps1:41:Write-Host "  Script:   $scriptPath"
scripts/win\install-rc-watchdog-task.ps1:42:Write-Host "  Actions:  recycle at 20h, kill zombies, Telegram alert"
scripts/win\install-rc-watchdog-task.ps1:43:Write-Host ""
scripts/win\install-rc-watchdog-task.ps1:44:Write-Host "To verify:"
scripts/win\install-rc-watchdog-task.ps1:45:Write-Host "  schtasks /Query /TN '$taskName' /V /FO LIST"
scripts/win\install-rc-watchdog-task.ps1:46:Write-Host ""
scripts/win\install-rc-watchdog-task.ps1:47:Write-Host "To test now:"
scripts/win\install-rc-watchdog-task.ps1:48:Write-Host "  schtasks /Run /TN '$taskName'"
scripts/win\install-rc-watchdog-task.ps1:49:Write-Host "  # or: powershell -File `"$scriptPath`""
scripts/win\install-rc-watchdog-task.ps1:50:Write-Host ""
scripts/win\install-research-task.ps1:1:# Install PF-AfterHoursResearch scheduled task
scripts/win\install-research-task.ps1:2:# Runs daily at 22:30 CET (after US market close)
scripts/win\install-research-task.ps1:3:# Uses Claude Opus to do deep research on markets and quant strategies
scripts/win\install-research-task.ps1:4:
scripts/win\install-research-task.ps1:5:$taskName = "PF-AfterHoursResearch"
scripts/win\install-research-task.ps1:6:$scriptPath = "Q:\finance-analyzer\scripts\after-hours-research.bat"
scripts/win\install-research-task.ps1:7:
scripts/win\install-research-task.ps1:8:# Remove existing task if present
scripts/win\install-research-task.ps1:9:$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
scripts/win\install-research-task.ps1:10:if ($existing) {
scripts/win\install-research-task.ps1:11:    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
scripts/win\install-research-task.ps1:12:    Write-Host "Removed existing task: $taskName"
scripts/win\install-research-task.ps1:13:}
scripts/win\install-research-task.ps1:14:
scripts/win\install-research-task.ps1:15:# Create the action
scripts/win\install-research-task.ps1:16:$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$scriptPath`"" -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-research-task.ps1:17:
scripts/win\install-research-task.ps1:18:# Trigger: daily at 22:30 (10:30 PM local time — after US close)
scripts/win\install-research-task.ps1:19:$trigger = New-ScheduledTaskTrigger -Daily -At "22:30"
scripts/win\install-research-task.ps1:20:
scripts/win\install-research-task.ps1:21:# Settings: allow long runs, restart on failure, run whether logged in or not
scripts/win\install-research-task.ps1:22:$settings = New-ScheduledTaskSettingsSet `
scripts/win\install-research-task.ps1:23:    -AllowStartIfOnBatteries `
scripts/win\install-research-task.ps1:24:    -DontStopIfGoingOnBatteries `
scripts/win\install-research-task.ps1:25:    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
scripts/win\install-research-task.ps1:26:    -RestartCount 1 `
scripts/win\install-research-task.ps1:27:    -RestartInterval (New-TimeSpan -Minutes 5) `
scripts/win\install-research-task.ps1:28:    -StartWhenAvailable
scripts/win\install-research-task.ps1:29:
scripts/win\install-research-task.ps1:30:# Register
scripts/win\install-research-task.ps1:31:Register-ScheduledTask `
scripts/win\install-research-task.ps1:32:    -TaskName $taskName `
scripts/win\install-research-task.ps1:33:    -Action $action `
scripts/win\install-research-task.ps1:34:    -Trigger $trigger `
scripts/win\install-research-task.ps1:35:    -Settings $settings `
scripts/win\install-research-task.ps1:36:    -Description "After-hours research agent: market review, quant research, signal audit, morning briefing" `
scripts/win\install-research-task.ps1:37:    -RunLevel Highest
scripts/win\install-research-task.ps1:38:
scripts/win\install-research-task.ps1:39:Write-Host ""
scripts/win\install-research-task.ps1:40:Write-Host "Installed: $taskName"
scripts/win\install-research-task.ps1:41:Write-Host "Schedule: Daily at 22:30 (after US market close)"
scripts/win\install-research-task.ps1:42:Write-Host "Script: $scriptPath"
scripts/win\install-research-task.ps1:43:Write-Host "Prompt: Q:\finance-analyzer\docs\after-hours-research-prompt.md"
scripts/win\install-research-task.ps1:44:Write-Host "Output: Q:\finance-analyzer\data\after-hours-research-out.txt"
scripts/win\install-research-task.ps1:45:Write-Host ""
scripts/win\install-research-task.ps1:46:Write-Host "To run manually: $scriptPath"
scripts/win\install-research-task.ps1:47:Write-Host "To check status: Get-ScheduledTask -TaskName $taskName"
scripts/win\install-signal-research-task.ps1:1:# Install PF-SignalResearch scheduled task
scripts/win\install-signal-research-task.ps1:2:# Run as: powershell -ExecutionPolicy Bypass -File scripts\win\install-signal-research-task.ps1
scripts/win\install-signal-research-task.ps1:3:#
scripts/win\install-signal-research-task.ps1:4:# Schedule:
scripts/win\install-signal-research-task.ps1:5:#   Daily at 18:30 CET (after EU market close, before after-hours research at 22:30)
scripts/win\install-signal-research-task.ps1:6:#   Runs Claude Code CLI with signal research prompt
scripts/win\install-signal-research-task.ps1:7:
scripts/win\install-signal-research-task.ps1:8:$taskName = "PF-SignalResearch"
scripts/win\install-signal-research-task.ps1:9:$scriptPath = "Q:\finance-analyzer\scripts\signal-research.bat"
scripts/win\install-signal-research-task.ps1:10:
scripts/win\install-signal-research-task.ps1:11:# Remove existing task if present
scripts/win\install-signal-research-task.ps1:12:$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
scripts/win\install-signal-research-task.ps1:13:if ($existing) {
scripts/win\install-signal-research-task.ps1:14:    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
scripts/win\install-signal-research-task.ps1:15:    Write-Host "Removed existing task: $taskName"
scripts/win\install-signal-research-task.ps1:16:}
scripts/win\install-signal-research-task.ps1:17:
scripts/win\install-signal-research-task.ps1:18:# Create the action
scripts/win\install-signal-research-task.ps1:19:$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$scriptPath`"" -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-signal-research-task.ps1:20:
scripts/win\install-signal-research-task.ps1:21:# Trigger: daily at 18:30 (after EU close, before after-hours at 22:30)
scripts/win\install-signal-research-task.ps1:22:$trigger = New-ScheduledTaskTrigger -Daily -At "18:30"
scripts/win\install-signal-research-task.ps1:23:
scripts/win\install-signal-research-task.ps1:24:# Settings: allow long runs, restart on failure, run whether logged in or not
scripts/win\install-signal-research-task.ps1:25:$settings = New-ScheduledTaskSettingsSet `
scripts/win\install-signal-research-task.ps1:26:    -AllowStartIfOnBatteries `
scripts/win\install-signal-research-task.ps1:27:    -DontStopIfGoingOnBatteries `
scripts/win\install-signal-research-task.ps1:28:    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
scripts/win\install-signal-research-task.ps1:29:    -RestartCount 1 `
scripts/win\install-signal-research-task.ps1:30:    -RestartInterval (New-TimeSpan -Minutes 5) `
scripts/win\install-signal-research-task.ps1:31:    -StartWhenAvailable
scripts/win\install-signal-research-task.ps1:32:
scripts/win\install-signal-research-task.ps1:33:# Register
scripts/win\install-signal-research-task.ps1:34:Register-ScheduledTask `
scripts/win\install-signal-research-task.ps1:35:    -TaskName $taskName `
scripts/win\install-signal-research-task.ps1:36:    -Action $action `
scripts/win\install-signal-research-task.ps1:37:    -Trigger $trigger `
scripts/win\install-signal-research-task.ps1:38:    -Settings $settings `
scripts/win\install-signal-research-task.ps1:39:    -Description "Daily AI signal research: academic papers, web search, scoring, implementation, backtest, codex review. Runs Claude Opus." `
scripts/win\install-signal-research-task.ps1:40:    -RunLevel Highest
scripts/win\install-signal-research-task.ps1:41:
scripts/win\install-signal-research-task.ps1:42:Write-Host ""
scripts/win\install-signal-research-task.ps1:43:Write-Host "Installed: $taskName"
scripts/win\install-signal-research-task.ps1:44:Write-Host "Schedule:  Daily at 18:30 (after EU close)"
scripts/win\install-signal-research-task.ps1:45:Write-Host "Script:    $scriptPath"
scripts/win\install-signal-research-task.ps1:46:Write-Host "Prompt:    Q:\finance-analyzer\docs\signal-research-prompt.md"
scripts/win\install-signal-research-task.ps1:47:Write-Host "Output:    Q:\finance-analyzer\data\signal_research_out.txt"
scripts/win\install-signal-research-task.ps1:48:Write-Host "Timeout:   2 hours max"
scripts/win\install-signal-research-task.ps1:49:Write-Host ""
scripts/win\install-signal-research-task.ps1:50:Write-Host "To run manually:  Start-ScheduledTask -TaskName '$taskName'"
scripts/win\install-signal-research-task.ps1:51:Write-Host "To check status:  Get-ScheduledTask -TaskName '$taskName'"
scripts/win\install-signal-research-task.ps1:52:Write-Host "To remove:        Unregister-ScheduledTask -TaskName '$taskName'"
scripts/win\meta-learner-retrain.bat:1:@echo off
scripts/win\meta-learner-retrain.bat:2:REM PF-MetaLearnerRetrain — Daily LightGBM meta-learner retraining (low priority)
scripts/win\meta-learner-retrain.bat:3:cd /d Q:\finance-analyzer
scripts/win\meta-learner-retrain.bat:4:set CLAUDECODE=
scripts/win\meta-learner-retrain.bat:5:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\meta-learner-retrain.bat:6:set PYTHONPATH=Q:\finance-analyzer
scripts/win\meta-learner-retrain.bat:7:
scripts/win\meta-learner-retrain.bat:8:echo [%date% %time%] Starting meta-learner retraining... >> data\meta_learner_retrain_out.txt 2>&1
scripts/win\meta-learner-retrain.bat:9:start /LOW /B .venv\Scripts\python.exe -u portfolio/meta_learner.py >> data\meta_learner_retrain_out.txt 2>&1
scripts/win\meta-learner-retrain.bat:10:echo [%date% %time%] Meta-learner retraining finished (code %ERRORLEVEL%). >> data\meta_learner_retrain_out.txt 2>&1
scripts/win\metals-arm-stop-once.bat:1:@echo off
scripts/win\metals-arm-stop-once.bat:2:cd /d Q:\finance-analyzer
scripts/win\metals-arm-stop-once.bat:3:.venv\Scripts\python.exe data\arm_stop_orders_once.py %*
scripts/win\metals-loop.bat:1:@echo off
scripts/win\metals-loop.bat:2:REM Metals Intraday Trading Loop — Layer 1 data collection + Claude Layer 2 decisions
scripts/win\metals-loop.bat:3:REM Auto-restarts on crash with 30s delay.
scripts/win\metals-loop.bat:4:cd /d Q:\finance-analyzer
scripts/win\metals-loop.bat:5:
scripts/win\metals-loop.bat:6::restart
scripts/win\metals-loop.bat:7:REM Clear Claude Code session markers so Layer 2 agent can launch
scripts/win\metals-loop.bat:8:set CLAUDECODE=
scripts/win\metals-loop.bat:9:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\metals-loop.bat:10:echo [%date% %time%] Starting metals loop...
scripts/win\metals-loop.bat:11:.venv\Scripts\python.exe -u data\metals_loop.py > data\metals_loop_out.txt 2>&1
scripts/win\metals-loop.bat:12:set EXIT_CODE=%ERRORLEVEL%
scripts/win\metals-loop.bat:13:echo [%date% %time%] Metals loop exited (code %EXIT_CODE%).
scripts/win\metals-loop.bat:14:
scripts/win\metals-loop.bat:15:REM Duplicate instance detected -- do not loop-restart into the active metals loop
scripts/win\metals-loop.bat:16:if %EXIT_CODE% EQU 11 (
scripts/win\metals-loop.bat:17:    echo [%date% %time%] Another metals loop instance already holds the lock -- stopping wrapper.
scripts/win\metals-loop.bat:18:    goto :eof
scripts/win\metals-loop.bat:19:)
scripts/win\metals-loop.bat:20:
scripts/win\metals-loop.bat:21:echo [%date% %time%] Restarting in 30s...
scripts/win\metals-loop.bat:22:timeout /t 30 /nobreak >nul
scripts/win\metals-loop.bat:23:goto restart
scripts/win\mstr-loop.bat:1:@echo off
scripts/win\mstr-loop.bat:2:REM MSTR Loop scheduled-task wrapper.
scripts/win\mstr-loop.bat:3:REM Usage: schtasks /run /tn "PF-MstrLoop"
scripts/win\mstr-loop.bat:4:REM Phase is read from MSTR_LOOP_PHASE env var, default "shadow".
scripts/win\mstr-loop.bat:5:
scripts/win\mstr-loop.bat:6:cd /d Q:\finance-analyzer
scripts/win\mstr-loop.bat:7:.venv\Scripts\python.exe -u -m portfolio.mstr_loop >> logs\mstr_loop_out.txt 2>&1
scripts/win\pf-agent.bat:1:@echo off
scripts/win\pf-agent.bat:2:REM Portfolio Intelligence — Claude Code Trading Agent (Layer 2)
scripts/win\pf-agent.bat:3:REM Invoked by Layer 1 (main.py) when a trigger fires, or manually.
scripts/win\pf-agent.bat:4:REM Claude Code auto-loads CLAUDE.md for project context. Trading playbook is in docs/TRADING_PLAYBOOK.md.
scripts/win\pf-agent.bat:5:
scripts/win\pf-agent.bat:6:cd /d Q:\finance-analyzer
scripts/win\pf-agent.bat:7:
scripts/win\pf-agent.bat:8:REM Clear Claude Code session markers — prevents "nested session" error when launched from
scripts/win\pf-agent.bat:9:REM a process tree that already has Claude Code running (e.g. Task Scheduler inheriting env)
scripts/win\pf-agent.bat:10:set CLAUDECODE=
scripts/win\pf-agent.bat:11:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\pf-agent.bat:12:
scripts/win\pf-agent.bat:13:REM Invoke Claude Code as the trading decision-maker
scripts/win\pf-agent.bat:14:REM Layer 1 already wrote fresh agent_summary.json — no need to re-collect
scripts/win\pf-agent.bat:15:echo Running trading agent...
scripts/win\pf-agent.bat:16:claude -p "You are the Layer 2 trading agent. FIRST read docs/TRADING_PLAYBOOK.md for trading rules. Then read data/layer2_context.md (your memory from previous invocations). Then read data/agent_summary_compact.json (signals, trigger reasons, timeframes), data/portfolio_state.json (Patient portfolio), and data/portfolio_state_bold.json (Bold portfolio). Follow the playbook to analyze, decide, and act for BOTH strategies independently. Compare your previous theses and prices with current data — were you right? Always write a journal entry and send a Telegram message." --allowedTools "Edit,Read,Bash,Write" --max-turns 40
scripts/win\oil-loop.bat:1:@echo off
scripts/win\oil-loop.bat:2:REM Oil Intraday Trading Loop — WTI paper-mode swing subsystem.
scripts/win\oil-loop.bat:3:REM Auto-restarts on crash with 30s delay. Exit code 11 means another
scripts/win\oil-loop.bat:4:REM instance already holds the singleton lock — we stop instead of
scripts/win\oil-loop.bat:5:REM fork-bombing into the live instance.
scripts/win\oil-loop.bat:6:cd /d Q:\finance-analyzer
scripts/win\oil-loop.bat:7:
scripts/win\oil-loop.bat:8::restart
scripts/win\oil-loop.bat:9:set CLAUDECODE=
scripts/win\oil-loop.bat:10:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\oil-loop.bat:11:echo [%date% %time%] Starting oil loop...
scripts/win\oil-loop.bat:12:.venv\Scripts\python.exe -u data\oil_loop.py --loop > data\oil_loop_out.txt 2>&1
scripts/win\oil-loop.bat:13:set EXIT_CODE=%ERRORLEVEL%
scripts/win\oil-loop.bat:14:echo [%date% %time%] Oil loop exited (code %EXIT_CODE%).
scripts/win\oil-loop.bat:15:
scripts/win\oil-loop.bat:16:if %EXIT_CODE% EQU 11 (
scripts/win\oil-loop.bat:17:    echo [%date% %time%] Another oil loop instance already holds the lock -- stopping wrapper.
scripts/win\oil-loop.bat:18:    goto :eof
scripts/win\oil-loop.bat:19:)
scripts/win\oil-loop.bat:20:
scripts/win\oil-loop.bat:21:echo [%date% %time%] Restarting in 30s...
scripts/win\oil-loop.bat:22:timeout /t 30 /nobreak >nul
scripts/win\oil-loop.bat:23:goto restart
scripts/win\pf-local-llm-report.bat:1:@echo off
scripts/win\pf-local-llm-report.bat:2:setlocal EnableExtensions EnableDelayedExpansion
scripts/win\pf-local-llm-report.bat:3:REM Standalone local-LLM report export. Safe to run outside the trading loop.
scripts/win\pf-local-llm-report.bat:4:
scripts/win\pf-local-llm-report.bat:5:for %%I in ("%~dp0..\..") do set REPO_ROOT=%%~fI
scripts/win\pf-local-llm-report.bat:6:cd /d "%REPO_ROOT%"
scripts/win\pf-local-llm-report.bat:7:
scripts/win\pf-local-llm-report.bat:8:set REPORT_DAYS=%~1
scripts/win\pf-local-llm-report.bat:9:if not defined REPORT_DAYS set REPORT_DAYS=30
scripts/win\pf-local-llm-report.bat:10:
scripts/win\pf-local-llm-report.bat:11:set LOG_FILE=%REPO_ROOT%\data\local_llm_report_task.log
scripts/win\pf-local-llm-report.bat:12:set PYTHON_EXE=%REPO_ROOT%\.venv\Scripts\python.exe
scripts/win\pf-local-llm-report.bat:13:if not exist "%PYTHON_EXE%" set PYTHON_EXE=Q:\finance-analyzer\.venv\Scripts\python.exe
scripts/win\pf-local-llm-report.bat:14:
scripts/win\pf-local-llm-report.bat:15:if not exist "%PYTHON_EXE%" (
scripts/win\pf-local-llm-report.bat:16:  echo [%date% %time%] Missing python executable: %PYTHON_EXE% >> "%LOG_FILE%"
scripts/win\pf-local-llm-report.bat:17:  exit /b 3
scripts/win\pf-local-llm-report.bat:18:)
scripts/win\pf-local-llm-report.bat:19:
scripts/win\pf-local-llm-report.bat:20:echo [%date% %time%] Starting local LLM report export (%REPORT_DAYS%d) >> "%LOG_FILE%"
scripts/win\pf-local-llm-report.bat:21:"%PYTHON_EXE%" portfolio\main.py --export-local-llm-report %REPORT_DAYS% >> "%LOG_FILE%" 2>&1
scripts/win\pf-local-llm-report.bat:22:set EXIT_CODE=%ERRORLEVEL%
scripts/win\pf-local-llm-report.bat:23:echo [%date% %time%] Finished local LLM report export exit=%EXIT_CODE% >> "%LOG_FILE%"
scripts/win\pf-local-llm-report.bat:24:
scripts/win\pf-local-llm-report.bat:25:exit /b %EXIT_CODE%
scripts/win\pf-llm-backfill.bat:1:@echo off
scripts/win\pf-llm-backfill.bat:2:REM PF-LLMBackfill scheduled task command.
scripts/win\pf-llm-backfill.bat:3:REM Runs the probability-log outcome backfill + sentiment A/B shadow backfill.
scripts/win\pf-llm-backfill.bat:4:REM Idempotent; rows without elapsed horizons are skipped and retried later.
scripts/win\pf-llm-backfill.bat:5:cd /d Q:\finance-analyzer
scripts/win\pf-llm-backfill.bat:6:.venv\Scripts\python.exe scripts\backfill_llm_outcomes.py >> data\llm_backfill_out.txt 2>&1
scripts/win\pf-llm-backfill.bat:7:.venv\Scripts\python.exe scripts\backfill_sentiment_shadow.py --horizon 1d >> data\llm_backfill_out.txt 2>&1
scripts/win\pf-outcome-check.bat:1:@echo off
scripts/win\pf-outcome-check.bat:2:REM PF-OutcomeCheck — Backfill price outcomes for signal accuracy tracking
scripts/win\pf-outcome-check.bat:3:cd /d Q:\finance-analyzer
scripts/win\pf-outcome-check.bat:4:set CLAUDECODE=
scripts/win\pf-outcome-check.bat:5:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\pf-outcome-check.bat:6:set PYTHONPATH=Q:\finance-analyzer
scripts/win\pf-outcome-check.bat:7:
scripts/win\pf-outcome-check.bat:8:echo [%date% %time%] Starting outcome backfill... >> data\outcome_check_out.txt 2>&1
scripts/win\pf-outcome-check.bat:9:.venv\Scripts\python.exe -u portfolio\main.py --check-outcomes >> data\outcome_check_out.txt 2>&1
scripts/win\pf-outcome-check.bat:10:echo [%date% %time%] Outcome backfill finished (code %ERRORLEVEL%). >> data\outcome_check_out.txt 2>&1
scripts/win\pf-loop.bat:1:@echo off
scripts/win\pf-loop.bat:2:REM Portfolio Intelligence — Continuous Loop (market-aware scheduling)
scripts/win\pf-loop.bat:3:REM Auto-restarts on crash with 30s delay.
scripts/win\pf-loop.bat:4:REM Uses START /WAIT to isolate Python in its own process group,
scripts/win\pf-loop.bat:5:REM preventing Ctrl+C from killing the restart loop.
scripts/win\pf-loop.bat:6:cd /d Q:\finance-analyzer
scripts/win\pf-loop.bat:7:
scripts/win\pf-loop.bat:8::restart
scripts/win\pf-loop.bat:9:REM Clear Claude Code session markers so Layer 2 agent can launch
scripts/win\pf-loop.bat:10:set CLAUDECODE=
scripts/win\pf-loop.bat:11:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\pf-loop.bat:12:set PYTHONPATH=Q:\finance-analyzer
scripts/win\pf-loop.bat:13:echo [%date% %time%] Starting loop...
scripts/win\pf-loop.bat:14:START /B /WAIT .venv\Scripts\python.exe -u portfolio\main.py --loop >> data\loop_out.txt 2>&1
scripts/win\pf-loop.bat:15:set EXIT_CODE=%ERRORLEVEL%
scripts/win\pf-loop.bat:16:echo [%date% %time%] Loop exited (code %EXIT_CODE%).
scripts/win\pf-loop.bat:17:
scripts/win\pf-loop.bat:18:REM Duplicate instance detected -- do not loop-restart into the active main loop
scripts/win\pf-loop.bat:19:if %EXIT_CODE% EQU 11 (
scripts/win\pf-loop.bat:20:    echo [%date% %time%] Another main loop instance already holds the lock -- stopping wrapper.
scripts/win\pf-loop.bat:21:    goto :eof
scripts/win\pf-loop.bat:22:)
scripts/win\pf-loop.bat:23:
scripts/win\pf-loop.bat:24:echo [%date% %time%] Restarting in 30s...
scripts/win\pf-loop.bat:25:timeout /t 30 /nobreak >nul
scripts/win\pf-loop.bat:26:goto restart
scripts/win\pf-loop-ensure.ps1:1:# pf-loop-ensure.ps1 — Idempotent loop launcher
scripts/win\pf-loop-ensure.ps1:2:# Safe to call on logon, wake-from-sleep, or manually.
scripts/win\pf-loop-ensure.ps1:3:# If pf-loop.bat is already running, exits silently.
scripts/win\pf-loop-ensure.ps1:4:
scripts/win\pf-loop-ensure.ps1:5:$batPath = "Q:\finance-analyzer\scripts\win\pf-loop.bat"
scripts/win\pf-loop-ensure.ps1:6:$lockFile = "Q:\finance-analyzer\data\pf-loop.pid"
scripts/win\pf-loop-ensure.ps1:7:
scripts/win\pf-loop-ensure.ps1:8:# Check if a pf-loop.bat process is already running
scripts/win\pf-loop-ensure.ps1:9:$existing = Get-Process cmd -ErrorAction SilentlyContinue |
scripts/win\pf-loop-ensure.ps1:10:    Where-Object { $_.MainWindowTitle -match 'pf-loop' -or $_.CommandLine -match 'pf-loop' }
scripts/win\pf-loop-ensure.ps1:11:
scripts/win\pf-loop-ensure.ps1:12:# Fallback: check if main.py --loop is running
scripts/win\pf-loop-ensure.ps1:13:if (-not $existing) {
scripts/win\pf-loop-ensure.ps1:14:    $pyLoop = Get-WmiObject Win32_Process -Filter "Name='python.exe'" |
scripts/win\pf-loop-ensure.ps1:15:        Where-Object { $_.CommandLine -match 'main\.py.*--loop' }
scripts/win\pf-loop-ensure.ps1:16:    if ($pyLoop) {
scripts/win\pf-loop-ensure.ps1:17:        Write-Host "[pf-loop-ensure] Loop already running (python PID $($pyLoop.ProcessId)). Exiting."
scripts/win\pf-loop-ensure.ps1:18:        exit 0
scripts/win\pf-loop-ensure.ps1:19:    }
scripts/win\pf-loop-ensure.ps1:20:}
scripts/win\pf-loop-ensure.ps1:21:
scripts/win\pf-loop-ensure.ps1:22:if ($existing) {
scripts/win\pf-loop-ensure.ps1:23:    Write-Host "[pf-loop-ensure] pf-loop.bat already running (PID $($existing.Id)). Exiting."
scripts/win\pf-loop-ensure.ps1:24:    exit 0
scripts/win\pf-loop-ensure.ps1:25:}
scripts/win\pf-loop-ensure.ps1:26:
scripts/win\pf-loop-ensure.ps1:27:Write-Host "[pf-loop-ensure] No loop detected. Starting pf-loop.bat..."
scripts/win\pf-loop-ensure.ps1:28:Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$batPath`"" -WindowStyle Minimized
scripts/win\pf-loop-ensure.ps1:29:Write-Host "[pf-loop-ensure] Launched pf-loop.bat at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
scripts/win\pf-restart.ps1:1:<#
scripts/win\pf-restart.ps1:2:.SYNOPSIS
scripts/win\pf-restart.ps1:3:  Cleanly restart a PF scheduled task (DataLoop, MetalsLoop, or both).
scripts/win\pf-restart.ps1:4:
scripts/win\pf-restart.ps1:5:.DESCRIPTION
scripts/win\pf-restart.ps1:6:  `schtasks /end <task>` only signals the wrapper batch script - the
scripts/win\pf-restart.ps1:7:  child python.exe spawned by `START /B /WAIT` survives as an orphan,
scripts/win\pf-restart.ps1:8:  keeping the old code in memory and blocking the file-singleton lock
scripts/win\pf-restart.ps1:9:  the new instance tries to acquire (exit code 11). Result: schtasks
scripts/win\pf-restart.ps1:10:  /run silently no-ops because the orphan still holds the resources
scripts/win\pf-restart.ps1:11:  the new wrapper would need.
scripts/win\pf-restart.ps1:12:
scripts/win\pf-restart.ps1:13:  This script does the right thing in one shot:
scripts/win\pf-restart.ps1:14:    1. Finds python.exe processes whose CommandLine matches the loop
scripts/win\pf-restart.ps1:15:       entry-point (portfolio.main --loop or portfolio.metals_loop).
scripts/win\pf-restart.ps1:16:    2. Stop-Process -Force on every matching PID.
scripts/win\pf-restart.ps1:17:    3. Calls schtasks /end + /run on the corresponding task name.
scripts/win\pf-restart.ps1:18:
scripts/win\pf-restart.ps1:19:.PARAMETER Target
scripts/win\pf-restart.ps1:20:  loop   - restart PF-DataLoop (main signal cycle)
scripts/win\pf-restart.ps1:21:  metals - restart PF-MetalsLoop (metals warrant trading)
scripts/win\pf-restart.ps1:22:  all    - restart both
scripts/win\pf-restart.ps1:23:  Default: loop.
scripts/win\pf-restart.ps1:24:
scripts/win\pf-restart.ps1:25:.EXAMPLE
scripts/win\pf-restart.ps1:26:  pwsh -File scripts/win/pf-restart.ps1
scripts/win\pf-restart.ps1:27:  pwsh -File scripts/win/pf-restart.ps1 -Target metals
scripts/win\pf-restart.ps1:28:  pwsh -File scripts/win/pf-restart.ps1 -Target all
scripts/win\pf-restart.ps1:29:#>
scripts/win\pf-restart.ps1:30:param(
scripts/win\pf-restart.ps1:31:    [ValidateSet('loop', 'metals', 'all')]
scripts/win\pf-restart.ps1:32:    [string]$Target = 'loop'
scripts/win\pf-restart.ps1:33:)
scripts/win\pf-restart.ps1:34:
scripts/win\pf-restart.ps1:35:$targets = @()
scripts/win\pf-restart.ps1:36:# Match patterns are regexes (NOT regex-escaped); they need to handle both
scripts/win\pf-restart.ps1:37:# the .venv launcher and the python312 actual interpreter, plus path
scripts/win\pf-restart.ps1:38:# separator variations. The `\\` in the patterns matches a literal backslash
scripts/win\pf-restart.ps1:39:# in the regex, which is what the live `tasklist` output shows.
scripts/win\pf-restart.ps1:40:switch ($Target) {
scripts/win\pf-restart.ps1:41:    'loop'   { $targets += @{Task='PF-DataLoop';   Match='main\.py.*--loop'} }
scripts/win\pf-restart.ps1:42:    'metals' { $targets += @{Task='PF-MetalsLoop'; Match='metals_loop\.py'} }
scripts/win\pf-restart.ps1:43:    'all'    {
scripts/win\pf-restart.ps1:44:        $targets += @{Task='PF-DataLoop';   Match='main\.py.*--loop'}
scripts/win\pf-restart.ps1:45:        $targets += @{Task='PF-MetalsLoop'; Match='metals_loop\.py'}
scripts/win\pf-restart.ps1:46:    }
scripts/win\pf-restart.ps1:47:}
scripts/win\pf-restart.ps1:48:
scripts/win\pf-restart.ps1:49:foreach ($t in $targets) {
scripts/win\pf-restart.ps1:50:    Write-Host "==> Restarting $($t.Task)" -ForegroundColor Cyan
scripts/win\pf-restart.ps1:51:
scripts/win\pf-restart.ps1:52:    # Find orphan-prone python processes by CommandLine match. The match
scripts/win\pf-restart.ps1:53:    # string is treated as a regex (already escaped where literal characters
scripts/win\pf-restart.ps1:54:    # need it).
scripts/win\pf-restart.ps1:55:    $procs = Get-CimInstance Win32_Process -Filter 'name="python.exe"' |
scripts/win\pf-restart.ps1:56:        Where-Object { $_.CommandLine -match $t.Match }
scripts/win\pf-restart.ps1:57:
scripts/win\pf-restart.ps1:58:    if ($procs) {
scripts/win\pf-restart.ps1:59:        foreach ($p in $procs) {
scripts/win\pf-restart.ps1:60:            Write-Host "    killing PID $($p.ProcessId) ($([datetime]$p.CreationDate))"
scripts/win\pf-restart.ps1:61:            try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop }
scripts/win\pf-restart.ps1:62:            catch { Write-Warning "    Stop-Process failed for $($p.ProcessId): $_" }
scripts/win\pf-restart.ps1:63:        }
scripts/win\pf-restart.ps1:64:        Start-Sleep -Seconds 2
scripts/win\pf-restart.ps1:65:    } else {
scripts/win\pf-restart.ps1:66:        Write-Host "    no running python.exe matching '$($t.Match)'"
scripts/win\pf-restart.ps1:67:    }
scripts/win\pf-restart.ps1:68:
scripts/win\pf-restart.ps1:69:    # Stop the scheduled task wrapper (cmd.exe running pf-*.bat). Safe even
scripts/win\pf-restart.ps1:70:    # if it's already gone - schtasks returns 1 in that case which we ignore.
scripts/win\pf-restart.ps1:71:    & schtasks.exe /end /tn $t.Task 2>&1 | Out-Null
scripts/win\pf-restart.ps1:72:
scripts/win\pf-restart.ps1:73:    # Start fresh
scripts/win\pf-restart.ps1:74:    & schtasks.exe /run /tn $t.Task
scripts/win\pf-restart.ps1:75:    if ($LASTEXITCODE -ne 0) {
scripts/win\pf-restart.ps1:76:        Write-Warning "    schtasks /run returned $LASTEXITCODE for $($t.Task)"
scripts/win\pf-restart.ps1:77:    }
scripts/win\pf-restart.ps1:78:}
scripts/win\pf-restart.ps1:79:
scripts/win\pf-restart.ps1:80:Write-Host ""
scripts/win\pf-restart.ps1:81:Write-Host "Verifying new processes..." -ForegroundColor Cyan
scripts/win\pf-restart.ps1:82:Start-Sleep -Seconds 5
scripts/win\pf-restart.ps1:83:foreach ($t in $targets) {
scripts/win\pf-restart.ps1:84:    $procs = Get-CimInstance Win32_Process -Filter 'name="python.exe"' |
scripts/win\pf-restart.ps1:85:        Where-Object { $_.CommandLine -match $t.Match }
scripts/win\pf-restart.ps1:86:    if ($procs) {
scripts/win\pf-restart.ps1:87:        foreach ($p in $procs) {
scripts/win\pf-restart.ps1:88:            Write-Host "  $($t.Task): PID $($p.ProcessId) started $([datetime]$p.CreationDate)" -ForegroundColor Green
scripts/win\pf-restart.ps1:89:        }
scripts/win\pf-restart.ps1:90:    } else {
scripts/win\pf-restart.ps1:91:        Write-Warning "  $($t.Task): no python.exe found yet - task may still be starting"
scripts/win\pf-restart.ps1:92:    }
scripts/win\pf-restart.ps1:93:}
scripts/win\pf-restart.bat:1:@echo off
scripts/win\pf-restart.bat:2:REM Thin wrapper around pf-restart.ps1 so the loop can be restarted from cmd.exe
scripts/win\pf-restart.bat:3:REM or `cmd.exe /c scripts\win\pf-restart.bat` from WSL. See pf-restart.ps1 for
scripts/win\pf-restart.bat:4:REM the full rationale on why schtasks /end leaves orphan python.exe children.
scripts/win\pf-restart.bat:5:REM
scripts/win\pf-restart.bat:6:REM Usage:
scripts/win\pf-restart.bat:7:REM   pf-restart.bat            (default: loop / PF-DataLoop)
scripts/win\pf-restart.bat:8:REM   pf-restart.bat metals     (PF-MetalsLoop only)
scripts/win\pf-restart.bat:9:REM   pf-restart.bat all        (both)
scripts/win\pf-restart.bat:10:
scripts/win\pf-restart.bat:11:set TARGET=%~1
scripts/win\pf-restart.bat:12:if "%TARGET%"=="" set TARGET=loop
scripts/win\pf-restart.bat:13:
scripts/win\pf-restart.bat:14:powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0pf-restart.ps1" -Target %TARGET%
scripts/win\pf-restart.bat:15:exit /b %ERRORLEVEL%
scripts/win\pf.bat:1:@echo off
scripts/win\pf.bat:2:cd /d Q:\finance-analyzer
scripts/win\pf.bat:3:.venv\Scripts\python.exe scripts\pf.py %*
scripts/win\pf-shadow-review.bat:1:@echo off
scripts/win\pf-shadow-review.bat:2:REM PF-ShadowReview scheduled task command.
scripts/win\pf-shadow-review.bat:3:REM Reports shadow signals older than 30 d without resolution.
scripts/win\pf-shadow-review.bat:4:REM Exit code 1 when stale shadows exist — Windows Task Scheduler treats
scripts/win\pf-shadow-review.bat:5:REM that as a failure so the task history surfaces the alert.
scripts/win\pf-shadow-review.bat:6:cd /d Q:\finance-analyzer
scripts/win\pf-shadow-review.bat:7:.venv\Scripts\python.exe scripts\review_shadow_signals.py >> data\shadow_review_out.txt 2>&1
scripts/win\rc-server-2.bat:1:@echo off
scripts/win\rc-server-2.bat:2:REM Claude Code Remote Control — Server 2 (Development)
scripts/win\rc-server-2.bat:3:cd /d Q:\finance-analyzer
scripts/win\rc-server-2.bat:4:set CLAUDECODE=
scripts/win\rc-server-2.bat:5:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\rc-server-2.bat:6:
scripts/win\rc-server-2.bat:7::restart
scripts/win\rc-server-2.bat:8:echo [%date% %time%] Starting RC server 2 (Development)... >> data\rc-server-2_out.txt 2>&1
scripts/win\rc-server-2.bat:9:claude remote-control --name "Development" --spawn worktree --capacity 4 >> data\rc-server-2_out.txt 2>&1
scripts/win\rc-server-2.bat:10:echo [%date% %time%] RC server 2 exited (code %ERRORLEVEL%). Restarting in 15s... >> data\rc-server-2_out.txt 2>&1
scripts/win\rc-server-2.bat:11:timeout /t 15 /nobreak >nul
scripts/win\rc-server-2.bat:12:goto restart
scripts/win\rc-keepalive.ps1:1:# rc-keepalive.ps1 - Soft-recycle idle RC servers to prevent Anthropic session delisting
scripts/win\rc-keepalive.ps1:2:#
scripts/win\rc-keepalive.ps1:3:# Problem: Anthropic server-side TTL is ~20 minutes. Only real user/model activity
scripts/win\rc-keepalive.ps1:4:# resets it - transport keepalives do NOT count. After ~20 min idle, the server deregisters
scripts/win\rc-keepalive.ps1:5:# the session. The CLI keeps polling (logs stay fresh), but the session becomes invisible
scripts/win\rc-keepalive.ps1:6:# in the claude.ai/code picker. Known bug: #28571, #29313, #34255, #37605, #38049.
scripts/win\rc-keepalive.ps1:7:#
scripts/win\rc-keepalive.ps1:8:# Solution: Every 5 min, check each RC server. If idle longer than its threshold
scripts/win\rc-keepalive.ps1:9:# (staggered: 13/15/17 min per server), kill it so the bat loop restarts fresh.
scripts/win\rc-keepalive.ps1:10:# This refreshes the Anthropic-side registration in ~15s.
scripts/win\rc-keepalive.ps1:11:#
scripts/win\rc-keepalive.ps1:12:# Staggered thresholds ensure at least one server is always fresh - they never all
scripts/win\rc-keepalive.ps1:13:# recycle at the same time after the first cycle.
scripts/win\rc-keepalive.ps1:14:#
scripts/win\rc-keepalive.ps1:15:# Servers with active work (>1 child session) are NEVER touched.
scripts/win\rc-keepalive.ps1:16:#
scripts/win\rc-keepalive.ps1:17:# Flags:
scripts/win\rc-keepalive.ps1:18:#   -Wake  Wake-from-sleep mode. Recycles ALL idle servers immediately (sleep guarantees
scripts/win\rc-keepalive.ps1:19:#          the 20-min TTL has expired, so all idle servers are stale).
scripts/win\rc-keepalive.ps1:20:#
scripts/win\rc-keepalive.ps1:21:# Designed to run via Task Scheduler every 5 min (PF-RCKeepalive).
scripts/win\rc-keepalive.ps1:22:
scripts/win\rc-keepalive.ps1:23:param(
scripts/win\rc-keepalive.ps1:24:    [switch]$Wake
scripts/win\rc-keepalive.ps1:25:)
scripts/win\rc-keepalive.ps1:26:
scripts/win\rc-keepalive.ps1:27:$logFile = "Q:\finance-analyzer\data\rc-keepalive.log"
scripts/win\rc-keepalive.ps1:28:$configFile = "C:\Users\Herc2\.config\finance-analyzer\config.json"
scripts/win\rc-keepalive.ps1:29:
scripts/win\rc-keepalive.ps1:30:# Staggered thresholds per server - keeps recycling naturally offset by ~2 min.
scripts/win\rc-keepalive.ps1:31:# All well under the ~20 min Anthropic TTL (4-8 min margin).
scripts/win\rc-keepalive.ps1:32:$serverThresholds = @{
scripts/win\rc-keepalive.ps1:33:    "Trading"     = 13
scripts/win\rc-keepalive.ps1:34:    "Development" = 15
scripts/win\rc-keepalive.ps1:35:    "Research"    = 17
scripts/win\rc-keepalive.ps1:36:}
scripts/win\rc-keepalive.ps1:37:$defaultThresholdMin = 15
scripts/win\rc-keepalive.ps1:38:
scripts/win\rc-keepalive.ps1:39:function Log($msg) {
scripts/win\rc-keepalive.ps1:40:    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
scripts/win\rc-keepalive.ps1:41:    $line = "[$ts] $msg"
scripts/win\rc-keepalive.ps1:42:    Write-Host $line
scripts/win\rc-keepalive.ps1:43:    Add-Content -Path $logFile -Value $line -ErrorAction SilentlyContinue
scripts/win\rc-keepalive.ps1:44:}
scripts/win\rc-keepalive.ps1:45:
scripts/win\rc-keepalive.ps1:46:function Send-Telegram($msg) {
scripts/win\rc-keepalive.ps1:47:    try {
scripts/win\rc-keepalive.ps1:48:        $cfg = Get-Content $configFile -Raw | ConvertFrom-Json
scripts/win\rc-keepalive.ps1:49:        $token  = $cfg.telegram.token
scripts/win\rc-keepalive.ps1:50:        $chatId = $cfg.telegram.chat_id
scripts/win\rc-keepalive.ps1:51:        if (-not $token -or -not $chatId) { return }
scripts/win\rc-keepalive.ps1:52:        $body = @{ chat_id = $chatId; text = $msg; parse_mode = "Markdown" } | ConvertTo-Json
scripts/win\rc-keepalive.ps1:53:        Invoke-RestMethod -Uri "https://api.telegram.org/bot$token/sendMessage" `
scripts/win\rc-keepalive.ps1:54:            -Method Post -ContentType "application/json" -Body $body `
scripts/win\rc-keepalive.ps1:55:            -TimeoutSec 15 | Out-Null
scripts/win\rc-keepalive.ps1:56:    } catch {
scripts/win\rc-keepalive.ps1:57:        Log "Telegram send failed: $_"
scripts/win\rc-keepalive.ps1:58:    }
scripts/win\rc-keepalive.ps1:59:}
scripts/win\rc-keepalive.ps1:60:
scripts/win\rc-keepalive.ps1:61:if ($Wake) {
scripts/win\rc-keepalive.ps1:62:    Log "Wake-from-sleep mode - all idle servers will be recycled immediately."
scripts/win\rc-keepalive.ps1:63:}
scripts/win\rc-keepalive.ps1:64:
scripts/win\rc-keepalive.ps1:65:# Get all claude.exe processes
scripts/win\rc-keepalive.ps1:66:$allClaude = Get-CimInstance Win32_Process -Filter "Name='claude.exe'" -ErrorAction SilentlyContinue
scripts/win\rc-keepalive.ps1:67:$rcProcs = $allClaude | Where-Object { $_.CommandLine -match 'remote-control' }
scripts/win\rc-keepalive.ps1:68:$childProcs = $allClaude | Where-Object { $_.CommandLine -match 'session-id' }
scripts/win\rc-keepalive.ps1:69:
scripts/win\rc-keepalive.ps1:70:if (-not $rcProcs) {
scripts/win\rc-keepalive.ps1:71:    Log "No RC servers running. Nothing to do."
scripts/win\rc-keepalive.ps1:72:    exit
scripts/win\rc-keepalive.ps1:73:}
scripts/win\rc-keepalive.ps1:74:
scripts/win\rc-keepalive.ps1:75:$recycled = 0
scripts/win\rc-keepalive.ps1:76:$skippedActive = 0
scripts/win\rc-keepalive.ps1:77:$skippedYoung = 0
scripts/win\rc-keepalive.ps1:78:$nameRx = '--name\s+"?(\w+)'
scripts/win\rc-keepalive.ps1:79:
scripts/win\rc-keepalive.ps1:80:foreach ($rc in $rcProcs) {
scripts/win\rc-keepalive.ps1:81:    $name = "Unknown"
scripts/win\rc-keepalive.ps1:82:    if ($rc.CommandLine -match $nameRx) { $name = $matches[1] }
scripts/win\rc-keepalive.ps1:83:    $ageMin = [math]::Round(((Get-Date) - $rc.CreationDate).TotalMinutes, 1)
scripts/win\rc-keepalive.ps1:84:    $children = @($childProcs | Where-Object { $_.ParentProcessId -eq $rc.ProcessId })
scripts/win\rc-keepalive.ps1:85:    $childCount = $children.Count
scripts/win\rc-keepalive.ps1:86:
scripts/win\rc-keepalive.ps1:87:    # Active sessions are NEVER touched, even on wake
scripts/win\rc-keepalive.ps1:88:    if ($childCount -gt 1) {
scripts/win\rc-keepalive.ps1:89:        Log "$name`: ${ageMin}m old, $childCount child sessions [ACTIVE WORK]. Protected."
scripts/win\rc-keepalive.ps1:90:        $skippedActive++
scripts/win\rc-keepalive.ps1:91:        continue
scripts/win\rc-keepalive.ps1:92:    }
scripts/win\rc-keepalive.ps1:93:
scripts/win\rc-keepalive.ps1:94:    # Determine threshold: 0 on wake (recycle everything idle), per-server otherwise
scripts/win\rc-keepalive.ps1:95:    $threshold = $defaultThresholdMin
scripts/win\rc-keepalive.ps1:96:    if ($Wake) {
scripts/win\rc-keepalive.ps1:97:        $threshold = 0
scripts/win\rc-keepalive.ps1:98:    } elseif ($serverThresholds.ContainsKey($name)) {
scripts/win\rc-keepalive.ps1:99:        $threshold = $serverThresholds[$name]
scripts/win\rc-keepalive.ps1:100:    }
scripts/win\rc-keepalive.ps1:101:
scripts/win\rc-keepalive.ps1:102:    if ($ageMin -lt $threshold) {
scripts/win\rc-keepalive.ps1:103:        Log "$name`: ${ageMin}m old, under ${threshold}m threshold. Fresh."
scripts/win\rc-keepalive.ps1:104:        $skippedYoung++
scripts/win\rc-keepalive.ps1:105:        continue
scripts/win\rc-keepalive.ps1:106:    }
scripts/win\rc-keepalive.ps1:107:
scripts/win\rc-keepalive.ps1:108:    # Old + idle = safe to recycle
scripts/win\rc-keepalive.ps1:109:    if ($Wake) {
scripts/win\rc-keepalive.ps1:110:        $reason = "wake-from-sleep"
scripts/win\rc-keepalive.ps1:111:    } else {
scripts/win\rc-keepalive.ps1:112:        $reason = "idle ${ageMin}m, threshold ${threshold}m"
scripts/win\rc-keepalive.ps1:113:    }
scripts/win\rc-keepalive.ps1:114:    Log "$name`: recycling [$reason]. $childCount child sessions."
scripts/win\rc-keepalive.ps1:115:    Stop-Process -Id $rc.ProcessId -Force -ErrorAction SilentlyContinue
scripts/win\rc-keepalive.ps1:116:    $recycled++
scripts/win\rc-keepalive.ps1:117:}
scripts/win\rc-keepalive.ps1:118:
scripts/win\rc-keepalive.ps1:119:if ($recycled -gt 0) {
scripts/win\rc-keepalive.ps1:120:    $total = @($rcProcs).Count
scripts/win\rc-keepalive.ps1:121:    if ($Wake) { $mode = "wake" } else { $mode = "periodic" }
scripts/win\rc-keepalive.ps1:122:    $summary = "Refreshed $recycled/$total idle servers [$mode]. $skippedActive active, $skippedYoung fresh."
scripts/win\rc-keepalive.ps1:123:    Log $summary
scripts/win\rc-keepalive.ps1:124:    Send-Telegram "*RC Keepalive* $summary"
scripts/win\rc-keepalive.ps1:125:} else {
scripts/win\rc-keepalive.ps1:126:    $total = @($rcProcs).Count
scripts/win\rc-keepalive.ps1:127:    Log "All $total servers OK: $skippedYoung fresh, $skippedActive active. No refresh needed."
scripts/win\rc-keepalive.ps1:128:}
scripts/win\rc-server-ensure.ps1:1:# rc-server-ensure.ps1 — Idempotent RC server launcher (3 independent servers)
scripts/win\rc-server-ensure.ps1:2:# Safe to call on logon, wake-from-sleep, or manually. Also runs every 30min via task repetition.
scripts/win\rc-server-ensure.ps1:3:#
scripts/win\rc-server-ensure.ps1:4:# Health check strategy (two-layer):
scripts/win\rc-server-ensure.ps1:5:#   1. Log heartbeat: each server writes "Reconnected" to its output file every 2-3 min.
scripts/win\rc-server-ensure.ps1:6:#      If the log file hasn't been modified in 10+ min, the server is stale — even if the
scripts/win\rc-server-ensure.ps1:7:#      process is alive and has a TCP socket open. This catches the case where a session
scripts/win\rc-server-ensure.ps1:8:#      looks healthy by TCP but is no longer registered with Anthropic's remote-control API.
scripts/win\rc-server-ensure.ps1:9:#   2. TCP fallback: if the log file doesn't exist yet (first launch), fall back to checking
scripts/win\rc-server-ensure.ps1:10:#      for an ESTABLISHED TCP connection to port 443.
scripts/win\rc-server-ensure.ps1:11:#
scripts/win\rc-server-ensure.ps1:12:# Design: sessions should live as long as the PC is on. Recycling kills context and can
scripts/win\rc-server-ensure.ps1:13:# leave spawned scripts/agents running unsupervised. Only recycle truly dead sessions.
scripts/win\rc-server-ensure.ps1:14:#
scripts/win\rc-server-ensure.ps1:15:# When called with -WakeDelay, waits 30s first to give servers time to reconnect after sleep.
scripts/win\rc-server-ensure.ps1:16:
scripts/win\rc-server-ensure.ps1:17:param(
scripts/win\rc-server-ensure.ps1:18:    [switch]$WakeDelay
scripts/win\rc-server-ensure.ps1:19:)
scripts/win\rc-server-ensure.ps1:20:
scripts/win\rc-server-ensure.ps1:21:$basePath = "Q:\finance-analyzer\scripts\win"
scripts/win\rc-server-ensure.ps1:22:$dataPath = "Q:\finance-analyzer\data"
scripts/win\rc-server-ensure.ps1:23:$logFile  = "$dataPath\rc-server-ensure.log"
scripts/win\rc-server-ensure.ps1:24:$staleMinutes = 10  # if log not updated in this many minutes, server is dead
scripts/win\rc-server-ensure.ps1:25:
scripts/win\rc-server-ensure.ps1:26:$servers = @(
scripts/win\rc-server-ensure.ps1:27:    @{ Name = "Trading";     Bat = "$basePath\rc-server.bat";   Pattern = '--name "?Trading';  Log = "$dataPath\rc-server_out.txt" },
scripts/win\rc-server-ensure.ps1:28:    @{ Name = "Development"; Bat = "$basePath\rc-server-2.bat"; Pattern = '--name "?Development'; Log = "$dataPath\rc-server-2_out.txt" },
scripts/win\rc-server-ensure.ps1:29:    @{ Name = "Research";    Bat = "$basePath\rc-server-3.bat"; Pattern = '--name "?Research'; Log = "$dataPath\rc-server-3_out.txt" }
scripts/win\rc-server-ensure.ps1:30:)
scripts/win\rc-server-ensure.ps1:31:
scripts/win\rc-server-ensure.ps1:32:$configFile = "C:\Users\Herc2\.config\finance-analyzer\config.json"
scripts/win\rc-server-ensure.ps1:33:
scripts/win\rc-server-ensure.ps1:34:function Log($msg) {
scripts/win\rc-server-ensure.ps1:35:    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
scripts/win\rc-server-ensure.ps1:36:    $line = "[$ts] $msg"
scripts/win\rc-server-ensure.ps1:37:    Write-Host $line
scripts/win\rc-server-ensure.ps1:38:    Add-Content -Path $logFile -Value $line -ErrorAction SilentlyContinue
scripts/win\rc-server-ensure.ps1:39:}
scripts/win\rc-server-ensure.ps1:40:
scripts/win\rc-server-ensure.ps1:41:function Send-Telegram($msg) {
scripts/win\rc-server-ensure.ps1:42:    try {
scripts/win\rc-server-ensure.ps1:43:        $cfg = Get-Content $configFile -Raw | ConvertFrom-Json
scripts/win\rc-server-ensure.ps1:44:        $token  = $cfg.telegram.token
scripts/win\rc-server-ensure.ps1:45:        $chatId = $cfg.telegram.chat_id
scripts/win\rc-server-ensure.ps1:46:        if (-not $token -or -not $chatId) { return }
scripts/win\rc-server-ensure.ps1:47:        $body = @{ chat_id = $chatId; text = $msg; parse_mode = "Markdown" } | ConvertTo-Json
scripts/win\rc-server-ensure.ps1:48:        Invoke-RestMethod -Uri "https://api.telegram.org/bot$token/sendMessage" `
scripts/win\rc-server-ensure.ps1:49:            -Method Post -ContentType "application/json" -Body $body `
scripts/win\rc-server-ensure.ps1:50:            -TimeoutSec 15 | Out-Null
scripts/win\rc-server-ensure.ps1:51:    } catch {
scripts/win\rc-server-ensure.ps1:52:        Log "Telegram send failed: $_"
scripts/win\rc-server-ensure.ps1:53:    }
scripts/win\rc-server-ensure.ps1:54:}
scripts/win\rc-server-ensure.ps1:55:
scripts/win\rc-server-ensure.ps1:56:# On wake-from-sleep, wait for servers to attempt their own reconnection
scripts/win\rc-server-ensure.ps1:57:# (built-in: 6 attempts over ~17s). 30s gives ample margin.
scripts/win\rc-server-ensure.ps1:58:if ($WakeDelay) {
scripts/win\rc-server-ensure.ps1:59:    Log "Wake-from-sleep trigger. Waiting 30s for auto-reconnect..."
scripts/win\rc-server-ensure.ps1:60:    Start-Sleep -Seconds 30
scripts/win\rc-server-ensure.ps1:61:}
scripts/win\rc-server-ensure.ps1:62:
scripts/win\rc-server-ensure.ps1:63:# Get all claude remote-control processes once
scripts/win\rc-server-ensure.ps1:64:$rcProcs = Get-CimInstance Win32_Process -Filter "Name='claude.exe'" -ErrorAction SilentlyContinue |
scripts/win\rc-server-ensure.ps1:65:    Where-Object { $_.CommandLine -match 'remote-control' }
scripts/win\rc-server-ensure.ps1:66:
scripts/win\rc-server-ensure.ps1:67:# Get all ESTABLISHED TCP connections to port 443 (Anthropic API) — used as fallback
scripts/win\rc-server-ensure.ps1:68:$tcpConns = Get-NetTCPConnection -State Established -RemotePort 443 -ErrorAction SilentlyContinue
scripts/win\rc-server-ensure.ps1:69:
scripts/win\rc-server-ensure.ps1:70:function Test-ServerAlive($srv, $procId) {
scripts/win\rc-server-ensure.ps1:71:    # Primary check: is the server's output log fresh?
scripts/win\rc-server-ensure.ps1:72:    # The RC server writes "Reconnected after Xs" every 2-3 minutes as part of its
scripts/win\rc-server-ensure.ps1:73:    # long-poll heartbeat. If the file hasn't been touched, the server is stuck.
scripts/win\rc-server-ensure.ps1:74:    $logPath = $srv.Log
scripts/win\rc-server-ensure.ps1:75:    if (Test-Path $logPath) {
scripts/win\rc-server-ensure.ps1:76:        $lastWrite = (Get-Item $logPath).LastWriteTime
scripts/win\rc-server-ensure.ps1:77:        $staleMins = [math]::Round(((Get-Date) - $lastWrite).TotalMinutes, 1)
scripts/win\rc-server-ensure.ps1:78:        if ($staleMins -lt $staleMinutes) {
scripts/win\rc-server-ensure.ps1:79:            return @{ Alive = $true; Method = "log"; Detail = "log updated ${staleMins}m ago" }
scripts/win\rc-server-ensure.ps1:80:        } else {
scripts/win\rc-server-ensure.ps1:81:            return @{ Alive = $false; Method = "log"; Detail = "log stale (${staleMins}m, threshold ${staleMinutes}m)" }
scripts/win\rc-server-ensure.ps1:82:        }
scripts/win\rc-server-ensure.ps1:83:    }
scripts/win\rc-server-ensure.ps1:84:
scripts/win\rc-server-ensure.ps1:85:    # Fallback: no log file yet (first launch). Check TCP connection.
scripts/win\rc-server-ensure.ps1:86:    $conn = $tcpConns | Where-Object { $_.OwningProcess -eq $procId }
scripts/win\rc-server-ensure.ps1:87:    if ($null -ne $conn -and @($conn).Count -gt 0) {
scripts/win\rc-server-ensure.ps1:88:        return @{ Alive = $true; Method = "tcp"; Detail = "ESTABLISHED to :443 (no log file yet)" }
scripts/win\rc-server-ensure.ps1:89:    }
scripts/win\rc-server-ensure.ps1:90:    return @{ Alive = $false; Method = "tcp"; Detail = "no connection and no log file" }
scripts/win\rc-server-ensure.ps1:91:}
scripts/win\rc-server-ensure.ps1:92:
scripts/win\rc-server-ensure.ps1:93:# Also find bat-loop cmd.exe processes (parents of the claude.exe servers)
scripts/win\rc-server-ensure.ps1:94:$batProcs = Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" -ErrorAction SilentlyContinue |
scripts/win\rc-server-ensure.ps1:95:    Where-Object { $_.CommandLine -match 'rc-server' }
scripts/win\rc-server-ensure.ps1:96:
scripts/win\rc-server-ensure.ps1:97:$launched = 0
scripts/win\rc-server-ensure.ps1:98:$skipped  = 0
scripts/win\rc-server-ensure.ps1:99:$killed   = 0
scripts/win\rc-server-ensure.ps1:100:
scripts/win\rc-server-ensure.ps1:101:foreach ($srv in $servers) {
scripts/win\rc-server-ensure.ps1:102:    $running = $rcProcs | Where-Object { $_.CommandLine -match $srv.Pattern }
scripts/win\rc-server-ensure.ps1:103:    $batName = [System.IO.Path]::GetFileName($srv.Bat)
scripts/win\rc-server-ensure.ps1:104:    $batRunning = $batProcs | Where-Object { $_.CommandLine -match [regex]::Escape($batName) }
scripts/win\rc-server-ensure.ps1:105:
scripts/win\rc-server-ensure.ps1:106:    if ($running) {
scripts/win\rc-server-ensure.ps1:107:        $procId = $running.ProcessId
scripts/win\rc-server-ensure.ps1:108:        $ageHrs = [math]::Round(((Get-Date) - $running.CreationDate).TotalHours, 1)
scripts/win\rc-server-ensure.ps1:109:        $check = Test-ServerAlive $srv $procId
scripts/win\rc-server-ensure.ps1:110:
scripts/win\rc-server-ensure.ps1:111:        if ($check.Alive) {
scripts/win\rc-server-ensure.ps1:112:            Log "$($srv.Name) healthy (PID $procId, ${ageHrs}h uptime, $($check.Detail)). Skipping."
scripts/win\rc-server-ensure.ps1:113:            $skipped++
scripts/win\rc-server-ensure.ps1:114:        } else {
scripts/win\rc-server-ensure.ps1:115:            # Dead: process alive but not actually working.
scripts/win\rc-server-ensure.ps1:116:            Log "$($srv.Name) dead (PID $procId, ${ageHrs}h uptime, $($check.Detail)). Killing; bat loop will restart."
scripts/win\rc-server-ensure.ps1:117:            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
scripts/win\rc-server-ensure.ps1:118:            $killed++
scripts/win\rc-server-ensure.ps1:119:            Send-Telegram "*RC Ensure* $($srv.Name) was dead after ${ageHrs}h ($($check.Detail)) -- recycled. Check if any spawned work was interrupted."
scripts/win\rc-server-ensure.ps1:120:            if (-not $batRunning) {
scripts/win\rc-server-ensure.ps1:121:                Log "$($srv.Name) bat loop also missing. Launching $($srv.Bat)..."
scripts/win\rc-server-ensure.ps1:122:                Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$($srv.Bat)`"" -WindowStyle Minimized
scripts/win\rc-server-ensure.ps1:123:                $launched++
scripts/win\rc-server-ensure.ps1:124:            }
scripts/win\rc-server-ensure.ps1:125:        }
scripts/win\rc-server-ensure.ps1:126:    } else {
scripts/win\rc-server-ensure.ps1:127:        if ($batRunning) {
scripts/win\rc-server-ensure.ps1:128:            # claude.exe died but bat loop is alive; it will restart on its own
scripts/win\rc-server-ensure.ps1:129:            Log "$($srv.Name) claude.exe gone but bat loop alive (PID $($batRunning.ProcessId)). Will auto-restart."
scripts/win\rc-server-ensure.ps1:130:            $skipped++
scripts/win\rc-server-ensure.ps1:131:        } else {
scripts/win\rc-server-ensure.ps1:132:            Log "$($srv.Name) not running. Launching $($srv.Bat)..."
scripts/win\rc-server-ensure.ps1:133:            Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$($srv.Bat)`"" -WindowStyle Minimized
scripts/win\rc-server-ensure.ps1:134:            $launched++
scripts/win\rc-server-ensure.ps1:135:        }
scripts/win\rc-server-ensure.ps1:136:    }
scripts/win\rc-server-ensure.ps1:137:}
scripts/win\rc-server-ensure.ps1:138:
scripts/win\rc-server-ensure.ps1:139:if ($launched -eq 0 -and $killed -eq 0) {
scripts/win\rc-server-ensure.ps1:140:    Log "All 3 RC servers healthy and connected."
scripts/win\rc-server-ensure.ps1:141:} else {
scripts/win\rc-server-ensure.ps1:142:    $summary = "Result: $skipped healthy, $killed dead recycled, $launched fresh launch(es)."
scripts/win\rc-server-ensure.ps1:143:    Log $summary
scripts/win\rc-server-ensure.ps1:144:    $trigger = if ($WakeDelay) { "wake-from-sleep" } else { "logon/periodic" }
scripts/win\rc-server-ensure.ps1:145:    Send-Telegram "*RC Ensure* ($trigger)`n$summary"
scripts/win\rc-server-ensure.ps1:146:}
scripts/win\rc-server-3.bat:1:@echo off
scripts/win\rc-server-3.bat:2:REM Claude Code Remote Control — Server 3 (Research)
scripts/win\rc-server-3.bat:3:cd /d Q:\finance-analyzer
scripts/win\rc-server-3.bat:4:set CLAUDECODE=
scripts/win\rc-server-3.bat:5:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\rc-server-3.bat:6:
scripts/win\rc-server-3.bat:7::restart
scripts/win\rc-server-3.bat:8:echo [%date% %time%] Starting RC server 3 (Research)... >> data\rc-server-3_out.txt 2>&1
scripts/win\rc-server-3.bat:9:claude remote-control --name "Research" --spawn worktree --capacity 4 >> data\rc-server-3_out.txt 2>&1
scripts/win\rc-server-3.bat:10:echo [%date% %time%] RC server 3 exited (code %ERRORLEVEL%). Restarting in 15s... >> data\rc-server-3_out.txt 2>&1
scripts/win\rc-server-3.bat:11:timeout /t 15 /nobreak >nul
scripts/win\rc-server-3.bat:12:goto restart
scripts/win\train-after-hours.bat:1:@echo off
scripts/win\train-after-hours.bat:2:REM Scheduled via Windows Task Scheduler at 22:30 CET daily
scripts/win\train-after-hours.bat:3:cd /d Q:\finance-analyzer
scripts/win\train-after-hours.bat:4:.venv\Scripts\python.exe -m portfolio.tinylora_trainer
scripts/win\rc-server.bat:1:@echo off
scripts/win\rc-server.bat:2:REM Claude Code Remote Control — Server 1 (Trading)
scripts/win\rc-server.bat:3:REM Auto-restarts on disconnect/crash with 15s delay.
scripts/win\rc-server.bat:4:REM Use rc-server-ensure.ps1 to launch all 3 servers independently.
scripts/win\rc-server.bat:5:cd /d Q:\finance-analyzer
scripts/win\rc-server.bat:6:
scripts/win\rc-server.bat:7:REM Clear Claude Code session markers (prevents nested-session errors)
scripts/win\rc-server.bat:8:set CLAUDECODE=
scripts/win\rc-server.bat:9:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\rc-server.bat:10:
scripts/win\rc-server.bat:11::restart
scripts/win\rc-server.bat:12:echo [%date% %time%] Starting RC server 1 (Trading)... >> data\rc-server_out.txt 2>&1
scripts/win\rc-server.bat:13:claude remote-control --name "Trading" --spawn worktree --capacity 4 >> data\rc-server_out.txt 2>&1
scripts/win\rc-server.bat:14:echo [%date% %time%] RC server 1 exited (code %ERRORLEVEL%). Restarting in 15s... >> data\rc-server_out.txt 2>&1
scripts/win\rc-server.bat:15:timeout /t 15 /nobreak >nul
scripts/win\rc-server.bat:16:goto restart
scripts/win\rc-watchdog.ps1:1:# rc-watchdog.ps1 - RC server health monitor + proactive recycle
scripts/win\rc-watchdog.ps1:2:# Runs on a schedule (every 30 min). Two jobs:
scripts/win\rc-watchdog.ps1:3:#   1. Proactive recycle: kill RC servers older than $MaxAgeHours so sessions
scripts/win\rc-watchdog.ps1:4:#      never hit the 24h server-side timeout. The bat loop auto-restarts.
scripts/win\rc-watchdog.ps1:5:#   2. Zombie detection: kill RC servers with no ESTABLISHED :443 connection.
scripts/win\rc-watchdog.ps1:6:# Sends Telegram alert on every action taken.
scripts/win\rc-watchdog.ps1:7:
scripts/win\rc-watchdog.ps1:8:param(
scripts/win\rc-watchdog.ps1:9:    [int]$MaxAgeHours = 20
scripts/win\rc-watchdog.ps1:10:)
scripts/win\rc-watchdog.ps1:11:
scripts/win\rc-watchdog.ps1:12:$ErrorActionPreference = "Continue"
scripts/win\rc-watchdog.ps1:13:$logFile  = "Q:\finance-analyzer\data\rc-watchdog.log"
scripts/win\rc-watchdog.ps1:14:$configFile = "C:\Users\Herc2\.config\finance-analyzer\config.json"
scripts/win\rc-watchdog.ps1:15:
scripts/win\rc-watchdog.ps1:16:$servers = @(
scripts/win\rc-watchdog.ps1:17:    @{ Name = "Trading";     BatFile = "rc-server.bat";   Pattern = '--name "?Trading' },
scripts/win\rc-watchdog.ps1:18:    @{ Name = "Development"; BatFile = "rc-server-2.bat"; Pattern = '--name "?Development' },
scripts/win\rc-watchdog.ps1:19:    @{ Name = "Research";    BatFile = "rc-server-3.bat"; Pattern = '--name "?Research' }
scripts/win\rc-watchdog.ps1:20:)
scripts/win\rc-watchdog.ps1:21:
scripts/win\rc-watchdog.ps1:22:function Log($msg) {
scripts/win\rc-watchdog.ps1:23:    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
scripts/win\rc-watchdog.ps1:24:    $line = "[$ts] $msg"
scripts/win\rc-watchdog.ps1:25:    Write-Host $line
scripts/win\rc-watchdog.ps1:26:    Add-Content -Path $logFile -Value $line -ErrorAction SilentlyContinue
scripts/win\rc-watchdog.ps1:27:}
scripts/win\rc-watchdog.ps1:28:
scripts/win\rc-watchdog.ps1:29:function Send-Telegram($msg) {
scripts/win\rc-watchdog.ps1:30:    try {
scripts/win\rc-watchdog.ps1:31:        $cfg = Get-Content $configFile -Raw | ConvertFrom-Json
scripts/win\rc-watchdog.ps1:32:        $token  = $cfg.telegram.token
scripts/win\rc-watchdog.ps1:33:        $chatId = $cfg.telegram.chat_id
scripts/win\rc-watchdog.ps1:34:        if (-not $token -or -not $chatId) {
scripts/win\rc-watchdog.ps1:35:            Log "Telegram: missing token or chat_id in config"
scripts/win\rc-watchdog.ps1:36:            return
scripts/win\rc-watchdog.ps1:37:        }
scripts/win\rc-watchdog.ps1:38:        $body = @{ chat_id = $chatId; text = $msg; parse_mode = "Markdown" } | ConvertTo-Json
scripts/win\rc-watchdog.ps1:39:        Invoke-RestMethod -Uri "https://api.telegram.org/bot$token/sendMessage" `
scripts/win\rc-watchdog.ps1:40:            -Method Post -ContentType "application/json" -Body $body `
scripts/win\rc-watchdog.ps1:41:            -TimeoutSec 15 | Out-Null
scripts/win\rc-watchdog.ps1:42:        Log "Telegram alert sent"
scripts/win\rc-watchdog.ps1:43:    } catch {
scripts/win\rc-watchdog.ps1:44:        Log "Telegram send failed: $_"
scripts/win\rc-watchdog.ps1:45:    }
scripts/win\rc-watchdog.ps1:46:}
scripts/win\rc-watchdog.ps1:47:
scripts/win\rc-watchdog.ps1:48:# --- Gather state ---
scripts/win\rc-watchdog.ps1:49:$rcProcs = Get-CimInstance Win32_Process -Filter "Name='claude.exe'" -ErrorAction SilentlyContinue |
scripts/win\rc-watchdog.ps1:50:    Where-Object { $_.CommandLine -match 'remote-control' }
scripts/win\rc-watchdog.ps1:51:
scripts/win\rc-watchdog.ps1:52:$tcpConns = Get-NetTCPConnection -State Established -RemotePort 443 -ErrorAction SilentlyContinue
scripts/win\rc-watchdog.ps1:53:
scripts/win\rc-watchdog.ps1:54:$batProcs = Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" -ErrorAction SilentlyContinue |
scripts/win\rc-watchdog.ps1:55:    Where-Object { $_.CommandLine -match 'rc-server' }
scripts/win\rc-watchdog.ps1:56:
scripts/win\rc-watchdog.ps1:57:function Test-Connected($procId) {
scripts/win\rc-watchdog.ps1:58:    $conn = $tcpConns | Where-Object { $_.OwningProcess -eq $procId }
scripts/win\rc-watchdog.ps1:59:    return ($null -ne $conn -and @($conn).Count -gt 0)
scripts/win\rc-watchdog.ps1:60:}
scripts/win\rc-watchdog.ps1:61:
scripts/win\rc-watchdog.ps1:62:$now = Get-Date
scripts/win\rc-watchdog.ps1:63:$actions = @()
scripts/win\rc-watchdog.ps1:64:$recycled = 0
scripts/win\rc-watchdog.ps1:65:$zombied  = 0
scripts/win\rc-watchdog.ps1:66:$healthy  = 0
scripts/win\rc-watchdog.ps1:67:$missing  = 0
scripts/win\rc-watchdog.ps1:68:$basePath = "Q:\finance-analyzer\scripts\win"
scripts/win\rc-watchdog.ps1:69:
scripts/win\rc-watchdog.ps1:70:foreach ($srv in $servers) {
scripts/win\rc-watchdog.ps1:71:    $proc = $rcProcs | Where-Object { $_.CommandLine -match $srv.Pattern }
scripts/win\rc-watchdog.ps1:72:    $bat  = $batProcs | Where-Object { $_.CommandLine -match [regex]::Escape($srv.BatFile) }
scripts/win\rc-watchdog.ps1:73:
scripts/win\rc-watchdog.ps1:74:    if (-not $proc) {
scripts/win\rc-watchdog.ps1:75:        if ($bat) {
scripts/win\rc-watchdog.ps1:76:            Log "$($srv.Name): claude.exe gone, bat loop alive - will auto-restart"
scripts/win\rc-watchdog.ps1:77:            $healthy++
scripts/win\rc-watchdog.ps1:78:        } else {
scripts/win\rc-watchdog.ps1:79:            Log "$($srv.Name): not running. Launching $($srv.BatFile)..."
scripts/win\rc-watchdog.ps1:80:            Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$basePath\$($srv.BatFile)`"" -WindowStyle Minimized
scripts/win\rc-watchdog.ps1:81:            $missing++
scripts/win\rc-watchdog.ps1:82:            $actions += "$($srv.Name): launched - was missing"
scripts/win\rc-watchdog.ps1:83:        }
scripts/win\rc-watchdog.ps1:84:        continue
scripts/win\rc-watchdog.ps1:85:    }
scripts/win\rc-watchdog.ps1:86:
scripts/win\rc-watchdog.ps1:87:    $pid_val = $proc.ProcessId
scripts/win\rc-watchdog.ps1:88:    $created = $proc.CreationDate
scripts/win\rc-watchdog.ps1:89:    $ageHours = ($now - $created).TotalHours
scripts/win\rc-watchdog.ps1:90:    $ageStr = "{0:N1}h" -f $ageHours
scripts/win\rc-watchdog.ps1:91:
scripts/win\rc-watchdog.ps1:92:    # Check 1: Proactive age-based recycle
scripts/win\rc-watchdog.ps1:93:    if ($ageHours -ge $MaxAgeHours) {
scripts/win\rc-watchdog.ps1:94:        Log "$($srv.Name): PID $pid_val age $ageStr >= ${MaxAgeHours}h threshold. Recycling..."
scripts/win\rc-watchdog.ps1:95:        Stop-Process -Id $pid_val -Force -ErrorAction SilentlyContinue
scripts/win\rc-watchdog.ps1:96:        $recycled++
scripts/win\rc-watchdog.ps1:97:        $msg = "$($srv.Name): recycled at $ageStr, limit ${MaxAgeHours}h"
scripts/win\rc-watchdog.ps1:98:
scripts/win\rc-watchdog.ps1:99:        if (-not $bat) {
scripts/win\rc-watchdog.ps1:100:            Log "$($srv.Name): bat loop missing after recycle. Launching..."
scripts/win\rc-watchdog.ps1:101:            Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$basePath\$($srv.BatFile)`"" -WindowStyle Minimized
scripts/win\rc-watchdog.ps1:102:            $msg += " + relaunched bat"
scripts/win\rc-watchdog.ps1:103:        }
scripts/win\rc-watchdog.ps1:104:        $actions += $msg
scripts/win\rc-watchdog.ps1:105:        continue
scripts/win\rc-watchdog.ps1:106:    }
scripts/win\rc-watchdog.ps1:107:
scripts/win\rc-watchdog.ps1:108:    # Check 2: Zombie detection (alive but no connection)
scripts/win\rc-watchdog.ps1:109:    if (-not (Test-Connected $pid_val)) {
scripts/win\rc-watchdog.ps1:110:        Log "$($srv.Name): PID $pid_val zombie - no :443 conn, age $ageStr. Killing..."
scripts/win\rc-watchdog.ps1:111:        Stop-Process -Id $pid_val -Force -ErrorAction SilentlyContinue
scripts/win\rc-watchdog.ps1:112:        $zombied++
scripts/win\rc-watchdog.ps1:113:        $msg = "$($srv.Name): killed zombie at $ageStr, no connection"
scripts/win\rc-watchdog.ps1:114:
scripts/win\rc-watchdog.ps1:115:        if (-not $bat) {
scripts/win\rc-watchdog.ps1:116:            Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$basePath\$($srv.BatFile)`"" -WindowStyle Minimized
scripts/win\rc-watchdog.ps1:117:            $msg += " + relaunched bat"
scripts/win\rc-watchdog.ps1:118:        }
scripts/win\rc-watchdog.ps1:119:        $actions += $msg
scripts/win\rc-watchdog.ps1:120:        continue
scripts/win\rc-watchdog.ps1:121:    }
scripts/win\rc-watchdog.ps1:122:
scripts/win\rc-watchdog.ps1:123:    # Healthy
scripts/win\rc-watchdog.ps1:124:    Log "$($srv.Name): healthy - PID $pid_val, connected, age $ageStr"
scripts/win\rc-watchdog.ps1:125:    $healthy++
scripts/win\rc-watchdog.ps1:126:}
scripts/win\rc-watchdog.ps1:127:
scripts/win\rc-watchdog.ps1:128:# --- Summary ---
scripts/win\rc-watchdog.ps1:129:if ($actions.Count -gt 0) {
scripts/win\rc-watchdog.ps1:130:    $time = Get-Date -Format 'HH:mm'
scripts/win\rc-watchdog.ps1:131:    $summary = "*RC Watchdog* $time`n"
scripts/win\rc-watchdog.ps1:132:    foreach ($a in $actions) {
scripts/win\rc-watchdog.ps1:133:        $summary += "- $a`n"
scripts/win\rc-watchdog.ps1:134:    }
scripts/win\rc-watchdog.ps1:135:    Log "Actions taken: $($actions -join '; ')"
scripts/win\rc-watchdog.ps1:136:    Send-Telegram $summary
scripts/win\rc-watchdog.ps1:137:} else {
scripts/win\rc-watchdog.ps1:138:    Log "All $healthy server(s) healthy. No action needed."
scripts/win\rc-watchdog.ps1:139:}
scripts/win\silver-monitor.bat:1:@echo off
scripts/win\silver-monitor.bat:2:title Silver Monitor (auto-restart)
scripts/win\silver-monitor.bat:3:cd /d Q:\finance-analyzer
scripts/win\silver-monitor.bat:4:
scripts/win\silver-monitor.bat:5::restart
scripts/win\silver-monitor.bat:6:echo [%date% %time%] Starting Silver Monitor...
scripts/win\silver-monitor.bat:7:.venv\Scripts\python.exe -u data\silver_monitor.py >> data\silver_monitor_out.txt 2>&1
scripts/win\silver-monitor.bat:8:set EXIT_CODE=%ERRORLEVEL%
scripts/win\silver-monitor.bat:9:echo [%date% %time%] Silver Monitor exited (code %EXIT_CODE%).
scripts/win\silver-monitor.bat:10:
scripts/win\silver-monitor.bat:11:REM Duplicate instance detected -- do not loop-restart into the active monitor
scripts/win\silver-monitor.bat:12:if %EXIT_CODE% EQU 11 (
scripts/win\silver-monitor.bat:13:    echo [%date% %time%] Another Silver Monitor instance already holds the lock -- stopping wrapper.
scripts/win\silver-monitor.bat:14:    goto :eof
scripts/win\silver-monitor.bat:15:)
scripts/win\silver-monitor.bat:16:
scripts/win\silver-monitor.bat:17:REM Check if within market hours (07:00-22:00 CET = 06:00-21:00 UTC)
scripts/win\silver-monitor.bat:18:REM If outside hours, exit instead of restarting
scripts/win\silver-monitor.bat:19:for /f "tokens=1-2 delims=:" %%a in ("%time: =0%") do set HOUR=%%a
scripts/win\silver-monitor.bat:20:if %HOUR% GEQ 22 (
scripts/win\silver-monitor.bat:21:    echo [%date% %time%] Outside market hours -- stopping.
scripts/win\silver-monitor.bat:22:    goto :eof
scripts/win\silver-monitor.bat:23:)
scripts/win\silver-monitor.bat:24:if %HOUR% LSS 7 (
scripts/win\silver-monitor.bat:25:    echo [%date% %time%] Outside market hours -- stopping.
scripts/win\silver-monitor.bat:26:    goto :eof
scripts/win\silver-monitor.bat:27:)
scripts/win\silver-monitor.bat:28:
scripts/win\silver-monitor.bat:29:echo [%date% %time%] Restarting in 30s...
scripts/win\silver-monitor.bat:30:timeout /t 30 /nobreak >nul
scripts/win\silver-monitor.bat:31:goto restart
scripts/win\settings-cleanup-20260508.bat:1:@echo off
scripts/win\settings-cleanup-20260508.bat:2:REM One-shot wrapper invoked by Windows Task Scheduler (PF-SettingsCleanup-20260508).
scripts/win\settings-cleanup-20260508.bat:3:REM Delegates to the WSL bash script that owns the actual cleanup logic.
scripts/win\settings-cleanup-20260508.bat:4:wsl.exe -d Ubuntu bash /mnt/q/finance-analyzer/scripts/cleanup_settings_20260508.sh
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 817ms:

[7m755:[0m@app.route("/")
[7m777:[0m@app.route("/logout")
[7m809:[0m@app.route("/api/summary")
[7m825:[0m@app.route("/api/signals")
[7m834:[0m@app.route("/api/portfolio")
[7m843:[0m@app.route("/api/portfolio-bold")
[7m852:[0m@app.route("/api/grid-fisher")
[7m875:[0m@app.route("/api/mstr_loop")
[7m922:[0m@app.route("/api/invocations")
[7m929:[0m@app.route("/api/telegrams")
[7m956:[0m@app.route("/api/signal-log")
[7m967:[0m@app.route("/api/accuracy")
[7m1047:[0m@app.route("/api/iskbets")
[7m1055:[0m@app.route("/api/lora-status")
[7m1067:[0m@app.route("/api/validate-portfolio", methods=["POST"])
[7m1095:[0m@app.route("/api/equity-curve")
[7m1110:[0m@app.route("/api/signal-heatmap")
[7m1209:[0m@app.route("/api/triggers")
[7m1217:[0m@app.route("/api/accuracy-history")
[7m1243:[0m@app.route("/api/local-llm-trends")
[7m1268:[0m@app.route("/api/metals-accuracy")
[7m1278:[0m@app.route("/api/trades")
[7m1309:[0m@app.route("/api/decisions")
[7m1352:[0m@app.route("/api/health")
[7m1368:[0m@app.route("/api/warrants")
[7m1385:[0m@app.route("/api/risk")
[7m1405:[0m@app.route("/api/metals")
[7m1462:[0m@app.route("/api/crypto")
[7m1493:[0m@app.route("/api/btc")
[7m1513:[0m@app.route("/api/eth")
[7m1533:[0m@app.route("/api/loop_health")
[7m1551:[0m@app.route("/api/oil")
[7m1589:[0m@app.route("/api/mstr")
[7m1615:[0m@app.route("/api/golddigger")
[7m1639:[0m@app.route("/api/market-health")
1[7m755:[0mdef _avanza_account_snapshot() -> dict:
1[7m777:[0m        "errors": ["avanza-worker: empty result"],
1[7m809:[0m    except Exception:
1[7m825:[0m
1[7m834:[0m        out["errors"].append(f"positions: {type(e).__name__}: {e}")
1[7m843:[0m        from portfolio.avanza_session import get_stop_losses
1[7m852:[0m
1[7m875:[0mdef _norm_stop(raw: dict) -> dict:
1[7m922:[0m# Tradeable assets — what the loops will buy/sell. Aggregates the metals
1[7m929:[0m@app.route("/api/tradeable_assets")
1[7m956:[0m        out["crypto_warrants"] = crypto.get("warrants", crypto) if isinstance(crypto, dict) else {}
1[7m967:[0m# ---------------------------------------------------------------------------

[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "''^" tests/conftest.py' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"claude|CLAUDECODE|CLAUDE_CODE_ENTRYPOINT|ExecutionPolicy|ErrorAction|schtasks|Start-Process|Stop-Process|Get-Content .*config|ConvertFrom-Json|Invoke-RestMethod|NoProfile|Bypass|cmd.exe|/c |python.exe|singleton|lock|orphan|taskkill|Get-ScheduledTask|Unregister-ScheduledTask|Register-ScheduledTask|New-ScheduledTaskAction|Set-ExecutionPolicy|Remove-Item|Move-Item|Write-Host|hardcoded|Q:\\\\finance-analyzer\" scripts/win/*.ps1 scripts/win/*.bat" in Q:\finance-analyzer
2026-05-11T15:29:43.675629Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'Get-ChildItem scripts/win -File -Include *.bat,*.ps1 | Select-Object -ExpandProperty FullName'` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'Get-ChildItem scripts/win -File -Include *.bat,*.ps1 | Select-Object -ExpandProperty FullName' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"logout|delete_cookie|set_cookie|clear_cookie|token=|force=|request\\.args|get_json\\(|jsonify\\(|send_from_directory\\(|send_file\\(|redirect\\(|make_response|require_auth|Authorization|Cf-Access|open\\(|json\\.loads\\(|json\\.load\\(\" dashboard/app.py" in Q:\finance-analyzer
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'Get-ChildItem scripts/win -File -Include *.bat,*.ps1 | Select-Object -ExpandProperty FullName'` rejected: blocked by policy
 succeeded in 797ms:
1:"""Shared pytest configuration and fixtures."""
2:
3:import json
4:
5:import numpy as np
6:import pandas as pd
7:import pytest
8:
9:
10:def pytest_configure(config):
11:    config.addinivalue_line(
12:        "markers", "integration: marks tests that require live GPU / network (deselect with '-k not integration')"
13:    )
14:
15:
16:@pytest.fixture(autouse=True)
17:def _reset_module_state():
18:    """Reset HIGH-risk module-level state before/after each test (xdist hygiene).
19:
20:    Module-level mutable globals (caches, process handles, counters) persist
21:    across test files sharded to the same xdist worker.  Without this fixture,
22:    test A can mutate ``signal_engine._adx_cache`` and test B — which expects
23:    an empty cache — fails when co-sharded but passes in isolation.
24:
25:    See: docs/IMPROVEMENT_BACKLOG.md TEST-HYGIENE-1
26:    """
27:    from _state_reset import reset_all_high_risk
28:    reset_all_high_risk()
29:    yield
30:    reset_all_high_risk()
31:
32:
33:@pytest.fixture(scope="session", autouse=True)
34:def _redirect_signal_utility_disk_cache(tmp_path_factory):
35:    """Redirect SIGNAL_UTILITY_CACHE_FILE to a session-scoped tmp path so the
36:    pytest suite NEVER touches the production data/signal_utility_cache.json.
37:
38:    Added 2026-05-04 after the L2 disk cache landed: invalidate_signal_utility_cache()
39:    now deletes the disk file, and the per-test in-memory clearing fixture below
40:    was indiscriminately wiping the production L2 cache every test run. With
41:    this session-level monkeypatch, the production file path is replaced once
42:    at session start with a tmpdir that gets cleaned up automatically when
43:    pytest exits.
44:
45:    Session-scoped because file path doesn't need per-test isolation (the
46:    in-memory clear below handles that), and a per-test redirect would burn
47:    a tmpdir per test for thousands of tests.
48:    """
49:    try:
50:        import portfolio.accuracy_stats as acc_mod
51:    except ImportError:
52:        yield
53:        return
54:    tmp_dir = tmp_path_factory.mktemp("signal_utility_disk")
55:    original = acc_mod.SIGNAL_UTILITY_CACHE_FILE
56:    acc_mod.SIGNAL_UTILITY_CACHE_FILE = tmp_dir / "signal_utility_cache.json"
57:    try:
58:        yield
59:    finally:
60:        acc_mod.SIGNAL_UTILITY_CACHE_FILE = original
61:
62:
63:@pytest.fixture(autouse=True)
64:def _isolate_signal_utility_cache():
65:    """Clear the signal_utility in-memory cache around each test.
66:
67:    Added 2026-04-15 alongside the BUG-178 TTL cache on signal_utility().
68:    Many existing tests in tests/test_signal_utility.py (and any future
69:    tests that do the same) patch portfolio.accuracy_stats.load_entries
70:    and then call signal_utility("1d") WITHOUT passing entries=
71:    explicitly. Before the cache landed, each such call went through
72:    load_entries and saw the patched data. After the cache landed the
73:    FIRST test in the worker populated _signal_utility_cache with its
74:    mocked result, and every subsequent test in the same worker read
75:    that stale value and silently ignored its own load_entries patch.
76:
77:    Clearing before AND after each test makes the fixture order-
78:    independent:
79:      - before: protects against state leaked from a prior test or
80:        module import
81:      - after: protects against this test's state leaking into others
82:
83:    2026-05-04: invalidate_signal_utility_cache() now also deletes the L2
84:    disk file. The session-scoped fixture above redirects the file path to
85:    a tmpdir, so per-test invalidate calls are safe (they delete the
86:    tmpdir's file, not production's).
87:    """
88:    try:
89:        from portfolio.accuracy_stats import invalidate_signal_utility_cache
90:    except ImportError:
91:        yield
92:        return
93:    invalidate_signal_utility_cache()
94:    try:
95:        yield
96:    finally:
97:        invalidate_signal_utility_cache()
98:
99:
100:# ---------------------------------------------------------------------------
101:# Indicator dictionary helpers
102:# ---------------------------------------------------------------------------
103:
104:def make_indicators(**overrides):
105:    """Return a baseline indicator dict suitable for most signal tests.
106:
107:    Override any key via keyword args::
108:
109:        ind = make_indicators(rsi=25.0, close=100_000.0)
110:    """
111:    base = {
112:        "close": 69_000.0,
113:        "rsi": 50.0,
114:        "macd_hist": 0.0,
115:        "macd_hist_prev": 0.0,
116:        "ema9": 69_000.0,
117:        "ema21": 69_000.0,
118:        "bb_upper": 70_000.0,
119:        "bb_lower": 68_000.0,
120:        "bb_mid": 69_000.0,
121:        "price_vs_bb": "inside",
122:        "atr": 1_500.0,
123:        "atr_pct": 2.2,
124:        "rsi_p20": 35.0,
125:        "rsi_p80": 65.0,
126:    }
127:    base.update(overrides)
128:    return base
129:
130:
131:# ---------------------------------------------------------------------------
132:# OHLCV DataFrame builders
133:# ---------------------------------------------------------------------------
134:
135:def make_candles(prices, volume=100.0):
136:    """Build a minimal OHLCV DataFrame from a list of close prices."""
137:    n = len(prices)
138:    return pd.DataFrame({
139:        "open": prices,
140:        "high": [p * 1.01 for p in prices],
141:        "low": [p * 0.99 for p in prices],
142:        "close": prices,
143:        "volume": [volume] * n,
144:        "time": pd.date_range("2026-01-01", periods=n, freq="15min"),
145:    })
146:
147:
148:def make_ohlcv_df(n=250, close_base=100.0, trend=0.0, volatility=1.0, seed=42):
149:    """Generate synthetic OHLCV data with configurable trend and volatility.
150:
151:    Useful for testing signals that need many bars (e.g., 200-SMA, Ichimoku).
152:    """
153:    rng = np.random.default_rng(seed)
154:    noise = rng.standard_normal(n) * volatility
155:    close = close_base + np.cumsum(noise) + np.arange(n) * trend
156:    close = np.maximum(close, 1.0)
157:    high = close + np.abs(rng.standard_normal(n) * volatility)
158:    low = close - np.abs(rng.standard_normal(n) * volatility)
159:    low = np.maximum(low, 0.5)
160:    opn = close + rng.standard_normal(n) * 0.3
161:    volume = rng.integers(100, 10_000, n).astype(float)
162:    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
163:    return pd.DataFrame({
164:        "open": opn, "high": high, "low": low,
165:        "close": close, "volume": volume, "time": dates,
166:    })
167:
168:
169:# ---------------------------------------------------------------------------
170:# Config fixtures
171:# ---------------------------------------------------------------------------
172:
173:@pytest.fixture
174:def sample_config():
175:    """Minimal config dict with telegram section (used by many modules)."""
176:    return {
177:        "telegram": {"token": "fake-token", "chat_id": "123456"},
178:    }
179:
180:
181:@pytest.fixture
182:def config_file(tmp_path, sample_config):
183:    """Write sample_config to a temp config.json and return its path."""
184:    cfg_path = tmp_path / "config.json"
185:    cfg_path.write_text(json.dumps(sample_config), encoding="utf-8")
186:    return cfg_path
187:
188:
189:# ---------------------------------------------------------------------------
190:# Temporary data directory
191:# ---------------------------------------------------------------------------
192:
193:@pytest.fixture
194:def tmp_data_dir(tmp_path):
195:    """Provide a temporary data directory (use with monkeypatch to override DATA_DIR)."""
196:    return tmp_path
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

2026-05-11T15:29:44.429442Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: scripts/win/*.ps1: The filename, directory name, or volume label syntax is incorrect. (os error 123)
rg: scripts/win/*.bat: The filename, directory name, or volume label syntax is incorrect. (os error 123)

 exited 1 in 813ms:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: scripts/win/*.ps1: The filename, directory name, or volume label syntax is incorrect. (os error 123)
rg: scripts/win/*.bat: The filename, directory name, or volume label syntax is incorrect. (os error 123)

 succeeded in 771ms:
14:from flask import Flask, jsonify, make_response, redirect, request, send_from_directory
58:    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
178:        value = int(request.args.get(name, default))
735:# so existing references (`require_auth`, `COOKIE_NAME`, etc.) keep working
737:# `from dashboard.app import require_auth` still resolves. Tests should
739:# take effect since require_auth resolves names via dashboard.auth's
747:    require_auth,
756:@require_auth
758:    # If the user arrived via ?token=XXX, the cookie was just set in
759:    # require_auth. Redirect to a token-less URL so the address bar (and
761:    # the Set-Cookie from require_auth's wrapped response.
762:    if request.args.get("token"):
763:        return redirect("/", code=302)
764:    return send_from_directory("static", "index.html")
768:@require_auth
772:    if request.args.get("token"):
773:        return redirect("/legacy", code=302)
774:    return send_from_directory("static", "index_legacy.html")
777:@app.route("/logout")
778:def logout():
787:    No `require_auth`: an unauthenticated visitor hitting /logout still gets
791:    response = redirect("/", code=302)
793:    response.set_cookie(
810:@require_auth
817:    return jsonify({
826:@require_auth
830:        return jsonify({"error": "no data"}), 404
831:    return jsonify(data)
835:@require_auth
839:        return jsonify({"error": "no data"}), 404
840:    return jsonify(data)
844:@require_auth
848:        return jsonify({"error": "no data"}), 404
849:    return jsonify(data)
853:@require_auth
872:    return jsonify({"state": state, "recent_decisions": decisions})
876:@require_auth
898:            with open(poll_path, encoding="utf-8") as f:
902:                            out["last_poll"] = _json.loads(line)
910:            with open(trades_path, encoding="utf-8") as f:
914:                            out["last_trade"] = _json.loads(line)
919:    return jsonify(out)
923:@require_auth
926:    return jsonify(entries)
930:@require_auth
940:    category_filter = request.args.get("category", "").strip().lower()
941:    search_filter = request.args.get("search", "").strip().lower()
953:    return jsonify(results)
957:@require_auth
960:    return jsonify(entries)
968:@require_auth
982:        return jsonify(_API_ACCURACY_CACHE["data"])
1041:        return jsonify(result)
1044:        return jsonify({"error": "Internal server error"}), 500
1048:@require_auth
1052:    return jsonify({"config": config, "state": state})
1056:@require_auth
1060:    return jsonify({"state": state, "training_progress": progress})
1068:@require_auth
1075:    data = request.get_json(silent=True)
1077:        return jsonify({"valid": False, "errors": ["No JSON body provided"]}), 400
1083:        return jsonify({"valid": False, "errors": [f"Validation error: {e}"]}), 500
1085:    return jsonify({
1096:@require_auth
1103:    return jsonify(entries)
1111:@require_auth
1119:        return jsonify({"error": "no data"}), 404
1194:    return jsonify({
1210:@require_auth
1214:    return jsonify(entries)
1218:@require_auth
1240:    return jsonify(entries)
1244:@require_auth
1253:    ticker = request.args.get("ticker", "").strip().upper() or None
1257:    return jsonify({
1269:@require_auth
1274:        return jsonify({"error": "no data", "stats": {}})
1275:    return jsonify(data)
1279:@require_auth
1306:    return jsonify(trades)
1310:@require_auth
1321:    ticker_filter = request.args.get("ticker", "").upper()
1322:    action_filter = request.args.get("action", "").upper()
1323:    strategy_filter = request.args.get("strategy", "").lower()
1349:    return jsonify(results)
1353:@require_auth
1358:        return jsonify(get_health_summary())
1361:        return jsonify({"error": "Internal server error"}), 500
1369:@require_auth
1377:        return jsonify({"holdings": {}, "transactions": []})
1378:    return jsonify(data)
1386:@require_auth
1394:        return jsonify({"monte_carlo": {}, "portfolio_var": {}})
1395:    return jsonify({
1406:@require_auth
1427:    return jsonify({
1463:@require_auth
1483:    return jsonify({
1494:@require_auth
1503:    return jsonify({
1514:@require_auth
1523:    return jsonify({
1534:@require_auth
1548:    return jsonify(read_loop_health())
1552:@require_auth
1578:    return jsonify({
1590:@require_auth
1603:    return jsonify({
1616:@require_auth
1628:    return jsonify({
1640:@require_auth
1669:        return jsonify(result)
1672:        return jsonify({"error": "Internal server error"}), 500
1893:@require_auth
1903:    `?force=1` bypasses the TTL cache so the user's manual Refresh
1907:    force = request.args.get("force", "").strip() in {"1", "true", "yes"}
1913:                return jsonify(cached)
1918:    return jsonify(snapshot)
1930:@require_auth
1964:    return jsonify(out)
1977:# cache mirrors the _AVANZA_CACHE pattern; ?force=1 bypasses for the
1982:@require_auth
1993:    fresher payload, and ``?force=1`` won't lose its refresh behind
1998:    force = request.args.get("force", "").strip() in {"1", "true", "yes"}
2003:                return jsonify(cached)
2008:            return jsonify(cached)
2012:        return jsonify(payload)
2016:@require_auth
2026:    force = request.args.get("force", "").strip() in {"1", "true", "yes"}
2031:                return jsonify(cached)
2035:            return jsonify(cached)
2039:        return jsonify(payload)
2045:# auth via dashboard.auth.require_auth. Path roots come from
2048:# House_blueprint imports `_get_config` and `require_auth` from
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

2026-05-11T15:30:03.697894Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "@'
from pathlib import Path
path = Path('dashboard/app.py')
for start,end in [(745,805),(870,920),(1060,1090),(1270,1310),(1305,1350)]:
    print(f'--- {start}-{end} ---')
    with path.open(encoding='utf-8') as f:
        for i,line in enumerate(f, start=1):
            if start <= i <= end:
                print(f'{i}:{line.rstrip()}')
            if i > end:
                break
'@ | python -"` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "@'
from pathlib import Path
path = Path('dashboard/app.py')
for start,end in [(745,805),(870,920),(1060,1090),(1270,1310),(1305,1350)]:
    print(f'--- {start}-{end} ---')
    with path.open(encoding='utf-8') as f:
        for i,line in enumerate(f, start=1):
            if start <= i <= end:
                print(f'{i}:{line.rstrip()}')
            if i > end:
                break
'@ | python -" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "@'
from pathlib import Path
path = Path('dashboard/app.py')
for start,end in [(1500,1665)]:
    print(f'--- {start}-{end} ---')
    with path.open(encoding='utf-8') as f:
        for i,line in enumerate(f, start=1):
            if start <= i <= end:
                print(f'{i}:{line.rstrip()}')
            if i > end:
                break
'@ | python -" in Q:\finance-analyzer
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "@'
from pathlib import Path
path = Path('dashboard/app.py')
for start,end in [(745,805),(870,920),(1060,1090),(1270,1310),(1305,1350)]:
    print(f'--- {start}-{end} ---')
    with path.open(encoding='utf-8') as f:
        for i,line in enumerate(f, start=1):
            if start <= i <= end:
                print(f'{i}:{line.rstrip()}')
            if i > end:
                break
'@ | python -"` rejected: blocked by policy
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "@'
from pathlib import Path
path = Path('dashboard/app.py')
for start,end in [(1500,1665)]:
    print(f'--- {start}-{end} ---')
    with path.open(encoding='utf-8') as f:
        for i,line in enumerate(f, start=1):
            if start <= i <= end:
                print(f'{i}:{line.rstrip()}')
            if i > end:
                break
'@ | python -"` rejected: blocked by policy
2026-05-11T15:30:03.698196Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "@'
from pathlib import Path
path = Path('dashboard/app.py')
for start,end in [(1500,1665)]:
    print(f'--- {start}-{end} ---')
    with path.open(encoding='utf-8') as f:
        for i,line in enumerate(f, start=1):
            if start <= i <= end:
                print(f'{i}:{line.rstrip()}')
            if i > end:
                break
'@ | python -"` rejected: blocked by policy
codex
I’m drilling into the few remaining hotspots with exact line ranges now: the dashboard routes that bypass the canonical readers, plus the Windows wrappers that interact with scheduled tasks and Claude subprocesses. After that I can collapse this into a strict severity-ordered review without padding.
2026-05-11T15:30:14.289829Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command '$i=1; Get-Content dashboard/app.py | ForEach-Object { if($i -ge 745 -and $i -le 805){ Write-Output ("{0}:{1}" -f $i, $_) }; $i++ }'` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command '$i=1; Get-Content dashboard/app.py | ForEach-Object { if($i -ge 745 -and $i -le 805){ Write-Output ("{0}:{1}" -f $i, $_) }; $i++ }' in Q:\finance-analyzer
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command '$i=1; Get-Content dashboard/app.py | ForEach-Object { if($i -ge 745 -and $i -le 805){ Write-Output ("{0}:{1}" -f $i, $_) }; $i++ }'` rejected: blocked by policy
2026-05-11T15:30:14.308605Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command '$i=1; Get-Content dashboard/app.py | ForEach-Object { if($i -ge 870 -and $i -le 920){ Write-Output ("{0}:{1}" -f $i, $_) }; $i++ }'` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command '$i=1; Get-Content dashboard/app.py | ForEach-Object { if($i -ge 870 -and $i -le 920){ Write-Output ("{0}:{1}" -f $i, $_) }; $i++ }' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command '$i=1; Get-Content dashboard/app.py | ForEach-Object { if($i -ge 1500 -and $i -le 1665){ Write-Output ("{0}:{1}" -f $i, $_) }; $i++ }' in Q:\finance-analyzer
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command '$i=1; Get-Content dashboard/app.py | ForEach-Object { if($i -ge 870 -and $i -le 920){ Write-Output ("{0}:{1}" -f $i, $_) }; $i++ }'` rejected: blocked by policy
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command '$i=1; Get-Content dashboard/app.py | ForEach-Object { if($i -ge 1500 -and $i -le 1665){ Write-Output ("{0}:{1}" -f $i, $_) }; $i++ }'` rejected: blocked by policy
2026-05-11T15:30:14.308942Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command '$i=1; Get-Content dashboard/app.py | ForEach-Object { if($i -ge 1500 -and $i -le 1665){ Write-Output ("{0}:{1}" -f $i, $_) }; $i++ }'` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 8 \"@app\\.route\\(\\\"/logout\\\"\\)|def logout|response\\.set_cookie|request\\.args\\.get\\(\\\"token\\\"\\)\" dashboard/app.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 8 \"@app\\.route\\(\\\"/api/mstr_loop\\\"\\)|def api_mstr_loop|with open\\(poll_path|with open\\(trades_path|_json\\.loads\\(line\\)\" dashboard/app.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 8 \"@app\\.route\\(\\\"/api/validate-portfolio\\\"|def api_validate_portfolio|request\\.get_json|validate_portfolio_data\" dashboard/app.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 6 \"@app\\.route\\(\\\"/api/trades\\\"\\)|def api_trades|return jsonify\\(trades\\)|_read_jsonl\\(\" dashboard/app.py" in Q:\finance-analyzer
2026-05-11T15:30:28.406278Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:@app\.route\(\)
    ^
error: unclosed group

 exited 1 in 816ms:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:@app\.route\(\)
    ^
error: unclosed group

2026-05-11T15:30:28.415576Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:@app\.route\(\)
    ^
error: unclosed group

 exited 1 in 798ms:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:@app\.route\(\)
    ^
error: unclosed group

2026-05-11T15:30:28.421688Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:@app\.route\(\)
    ^
error: unclosed group

 exited 1 in 826ms:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:@app\.route\(\)
    ^
error: unclosed group

2026-05-11T15:30:28.430225Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:@app\.route\(\)
    ^
error: unclosed group

 exited 1 in 812ms:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:@app\.route\(\)
    ^
error: unclosed group

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 8 '@app\\.route\\(\"/logout\"\\)|def logout|response\\.set_cookie|request\\.args\\.get\\(\"token\"\\)' dashboard/app.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 8 '@app\\.route\\(\"/api/mstr_loop\"\\)|def api_mstr_loop|with open\\(poll_path|with open\\(trades_path|_json\\.loads\\(line\\)' dashboard/app.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 8 '@app\\.route\\(\"/api/validate-portfolio\"|def api_validate_portfolio|request\\.get_json|validate_portfolio_data' dashboard/app.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 6 '@app\\.route\\(\"/api/trades\"\\)|def api_trades|return jsonify\\(trades\\)|_read_jsonl\\(' dashboard/app.py" in Q:\finance-analyzer
 succeeded in 680ms:
754-
755-@app.route("/")
756-@require_auth
757-def index():
758-    # If the user arrived via ?token=XXX, the cookie was just set in
759-    # require_auth. Redirect to a token-less URL so the address bar (and
760-    # whatever the user bookmarks next) stays clean. The redirect inherits
761-    # the Set-Cookie from require_auth's wrapped response.
762:    if request.args.get("token"):
763-        return redirect("/", code=302)
764-    return send_from_directory("static", "index.html")
765-
766-
767-@app.route("/legacy")
768-@require_auth
769-def index_legacy():
770-    # Pre-redesign single-file dashboard preserved as a fallback during the
771-    # 2026-05-03 mobile-first rollout. See docs/PLAN.md.
772:    if request.args.get("token"):
773-        return redirect("/legacy", code=302)
774-    return send_from_directory("static", "index_legacy.html")
775-
776-
777:@app.route("/logout")
778:def logout():
779-    """Clear the pf_dashboard_token cookie and redirect to /.
780-
781-    The auth cookie is HttpOnly, so client JS cannot expire it via
782-    document.cookie — the browser ignores any attempt to write a name that
783-    Set-Cookie marked HttpOnly. The mobile Settings → Sign out button
784-    therefore has to navigate here so the server can emit the matching
785-    Set-Cookie with Max-Age=0. (Codex P2 finding 2026-05-03.)
786-
787-    No `require_auth`: an unauthenticated visitor hitting /logout still gets
788-    the cookie wiped (harmless — they had no valid cookie anyway) and
789-    Cloudflare Access still gates the redirected destination.
790-    """
791-    response = redirect("/", code=302)
792-    # Match every flag we set on the original cookie except expiry.
793:    response.set_cookie(
794-        "pf_dashboard_token",
795-        "",
796-        max_age=0,
797-        expires=0,
798-        httponly=True,
799-        secure=True,
800-        samesite="Lax",
801-    )
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 675ms:
99-# ---------------------------------------------------------------------------
100-
101-def _read_json(path, ttl=_DEFAULT_TTL):
102-    return _cached_read(f"json:{path}", ttl, lambda: _load_json_impl(path))
103-
104-
105:def _read_jsonl(path, limit=100, ttl=_DEFAULT_TTL):
106-    """Cached JSONL read returning the last `limit` entries.
107-
108-    Switched from load_jsonl(limit=) (full scan + deque) to
109-    load_jsonl_tail (seek from end). For an 80MB log the difference is
110-    ~880ms vs ~5ms.
111-
--
180-        value = default
181-    return max(1, min(value, max_value))
182-
183-
184-def _iter_latest_dict_entries(path, read_limit):
185-    """Yield JSONL entries newest-first, skipping non-dict shapes."""
186:    raw = _read_jsonl(path, limit=read_limit)
187-    for entry in reversed(raw):
188-        if isinstance(entry, dict):
189-            yield entry
190-
191-
192-def _parse_iso8601(value):
--
499-    return primary
500-
501-
502-def _build_metals_context_fallback(decisions):
503-    positions_state = _read_json(DATA_DIR / "metals_positions_state.json") or {}
504-    signal_entries = list(_iter_latest_dict_entries(DATA_DIR / "metals_signal_log.jsonl", read_limit=10))
505:    value_history = _read_jsonl(DATA_DIR / "metals_value_history.jsonl", limit=10)
506-    technicals = _read_json(DATA_DIR / "silver_analysis.json") or {}
507-    latest_signal = signal_entries[0] if signal_entries else {}
508-    latest_value = value_history[-1] if value_history else {}
509-    latest_decision = decisions[0] if decisions else {}
510-
511-    if not positions_state and not latest_signal and not latest_value and not latest_decision:
--
919-    return jsonify(out)
920-
921-
922-@app.route("/api/invocations")
923-@require_auth
924-def api_invocations():
925:    entries = _read_jsonl(DATA_DIR / "invocations.jsonl", limit=50)
926-    return jsonify(entries)
927-
928-
929-@app.route("/api/telegrams")
930-@require_auth
931-def api_telegrams():
--
953-    return jsonify(results)
954-
955-
956-@app.route("/api/signal-log")
957-@require_auth
958-def api_signal_log():
959:    entries = _read_jsonl(DATA_DIR / "signal_log.jsonl", limit=50)
960-    return jsonify(entries)
961-
962-
963-_API_ACCURACY_CACHE: dict = {"ts": 0.0, "data": None}
964-_API_ACCURACY_TTL_SEC = 60.0
965-
--
1096-@require_auth
1097-def api_equity_curve():
1098-    """Return portfolio value history for charting.
1099-
1100-    Reads data/portfolio_value_history.jsonl. Returns empty array if missing.
1101-    """
1102:    entries = _read_jsonl(DATA_DIR / "portfolio_value_history.jsonl", limit=5000)
1103-    return jsonify(entries)
1104-
1105-
1106-# ---------------------------------------------------------------------------
1107-# New: Signal heatmap (30 signals x all tickers)
1108-# ---------------------------------------------------------------------------
--
1207-# ---------------------------------------------------------------------------
1208-
1209-@app.route("/api/triggers")
1210-@require_auth
1211-def api_triggers():
1212-    """Return last 50 trigger/invocation events from invocations.jsonl."""
1213:    entries = _read_jsonl(DATA_DIR / "invocations.jsonl", limit=50)
1214-    return jsonify(entries)
1215-
1216-
1217-@app.route("/api/accuracy-history")
1218-@require_auth
1219-def api_accuracy_history():
--
1221-
1222-    2026-05-05: tag each per-signal slice with `enabled` so the chart
1223-    can dim/exclude force-HOLD'd signals. Tag is derived at response
1224-    time from DISABLED_SIGNALS so historical snapshots written before
1225-    the flag existed are also tagged correctly.
1226-    """
1227:    entries = _read_jsonl(DATA_DIR / "accuracy_snapshots.jsonl", limit=500)
1228-    try:
1229-        from portfolio.tickers import DISABLED_SIGNALS
1230-        for snap in entries:
1231-            sigs = snap.get("signals") if isinstance(snap, dict) else None
1232-            if not isinstance(sigs, dict):
1233-                continue
--
1249-      - limit: number of history points to return (default 90, max 366)
1250-      - ticker: optional ticker filter for Ministral per-ticker series
1251-    """
1252-    limit = _parse_limit_arg("limit", default=90, max_value=366)
1253-    ticker = request.args.get("ticker", "").strip().upper() or None
1254-    latest = _read_json(DATA_DIR / "local_llm_report_latest.json")
1255:    history = _read_jsonl(DATA_DIR / "local_llm_report_history.jsonl", limit=limit)
1256-
1257-    return jsonify({
1258-        "ticker": ticker,
1259-        "latest": latest,
1260-        "series": [
1261-            _build_local_llm_trend_point(entry, ticker=ticker)
--
1272-    data = _read_json(DATA_DIR / "metals_signal_accuracy.json")
1273-    if not data:
1274-        return jsonify({"error": "no data", "stats": {}})
1275-    return jsonify(data)
1276-
1277-
1278:@app.route("/api/trades")
1279-@require_auth
1280:def api_trades():
1281-    """Return combined transactions from both portfolio states for chart annotations."""
1282-    patient = _read_json(DATA_DIR / "portfolio_state.json")
1283-    bold = _read_json(DATA_DIR / "portfolio_state_bold.json")
1284-    trades = []
1285-    if patient and patient.get("transactions"):
1286-        for tx in patient["transactions"]:
--
1300-                "action": tx.get("action", ""),
1301-                "total_sek": tx.get("total_sek", 0),
1302-                "price_usd": tx.get("price_usd", 0),
1303-                "strategy": "bold",
1304-            })
1305-    trades.sort(key=lambda t: t.get("ts", ""))
1306:    return jsonify(trades)
1307-
1308-
1309-@app.route("/api/decisions")
1310-@require_auth
1311-def api_decisions():
1312-    """Return Layer 2 decision history with optional filtering.
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 677ms:
1059-    progress = _read_json(TRAINING_DIR / "training_progress.json")
1060-    return jsonify({"state": state, "training_progress": progress})
1061-
1062-
1063-# ---------------------------------------------------------------------------
1064-# New: Portfolio validation
1065-# ---------------------------------------------------------------------------
1066-
1067:@app.route("/api/validate-portfolio", methods=["POST"])
1068-@require_auth
1069:def api_validate_portfolio():
1070-    """Validate a portfolio JSON for integrity.
1071-
1072-    Delegates to portfolio_validator.validate_portfolio() which performs
1073-    comprehensive checks: cash, holdings, fees, transactions, avg_cost.
1074-    """
1075:    data = request.get_json(silent=True)
1076-    if not data:
1077-        return jsonify({"valid": False, "errors": ["No JSON body provided"]}), 400
1078-
1079-    try:
1080-        from portfolio.portfolio_validator import validate_portfolio
1081-        errors = validate_portfolio(data)
1082-    except Exception as e:
1083-        return jsonify({"valid": False, "errors": [f"Validation error: {e}"]}), 500
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 703ms:
867-        if decisions_path.exists():
868-            from portfolio.file_utils import load_jsonl_tail
869-            decisions = load_jsonl_tail(decisions_path, max_entries=50)
870-    except Exception:
871-        decisions = []
872-    return jsonify({"state": state, "recent_decisions": decisions})
873-
874-
875:@app.route("/api/mstr_loop")
876-@require_auth
877:def api_mstr_loop():
878-    """Live snapshot of the MSTR Loop bot (v2 Tier 3).
879-
880-    Returns state + scorecard + latest poll in one JSON:
881-        {
882-          "state": {cash, positions, total_pnl, ...},
883-          "scorecard": {win_rate, expectancy, trades_by_strategy, ...},
884-          "last_poll": {last cycle snapshot from mstr_loop_poll.jsonl},
885-          "last_trade": {last closed trade from mstr_loop_trades.jsonl},
--
890-        "scorecard": _read_json(DATA_DIR / "mstr_loop_scorecard.json") or {},
891-        "last_poll": None,
892-        "last_trade": None,
893-    }
894-    import json as _json
895-    poll_path = DATA_DIR / "mstr_loop_poll.jsonl"
896-    if poll_path.exists():
897-        try:
898:            with open(poll_path, encoding="utf-8") as f:
899-                for line in f:
900-                    if line.strip():
901-                        try:
902:                            out["last_poll"] = _json.loads(line)
903-                        except _json.JSONDecodeError:
904-                            pass
905-        except OSError:
906-            pass
907-    trades_path = DATA_DIR / "mstr_loop_trades.jsonl"
908-    if trades_path.exists():
909-        try:
910:            with open(trades_path, encoding="utf-8") as f:
911-                for line in f:
912-                    if line.strip():
913-                        try:
914:                            out["last_trade"] = _json.loads(line)
915-                        except _json.JSONDecodeError:
916-                            pass
917-        except OSError:
918-            pass
919-    return jsonify(out)
920-
921-
922-@app.route("/api/invocations")
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"acquire_lock_file\\(|release_lock_file\\(|singleton\\.lock|EXIT_CODE% EQU 11|return 11|sys\\.exit\\(11\\)|exit\\(11\\)\" portfolio scripts dashboard" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"atomic_write_json\\(|atomic_append_jsonl\\(|atomic_write_jsonl\\(|prune_jsonl\\(|jsonl_sidecar_lock\\(|load_json\\(|require_json\\(|load_jsonl\\(|load_jsonl_tail\\(\" portfolio dashboard scripts" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"ProcessLock|singleton lock|singleton\\.lock|acquire_lock_file\\(\" portfolio/process_lock.py scripts/win portfolio dashboard" in Q:\finance-analyzer
 succeeded in 740ms:
dashboard\trading_status.py:86:    state = load_json(dd / "golddigger_state.json", default=None)
dashboard\trading_status.py:118:    state = load_json(dd / "elongir_state.json", default=None)
dashboard\trading_status.py:150:    state = load_json(dd / "metals_swing_state.json", default=None)
dashboard\trading_status.py:151:    guard = load_json(dd / "metals_guard_state.json", default={}) or {}
dashboard\trading_status.py:183:    state = load_json(dd / "fish_engine_state.json", default=None)
dashboard\system_status.py:90:        health = load_json(dd / "health_state.json", default={}) or {}
dashboard\system_status.py:138:        entries = load_jsonl(dd / "critical_errors.jsonl")
dashboard\system_status.py:277:        entries = load_jsonl(dd / "critical_errors.jsonl")
dashboard\system_status.py:319:        tail = load_jsonl_tail(dd / "layer2_journal.jsonl", max_entries=20)
dashboard\system_status.py:423:        health = load_json(dd / "health_state.json", default={}) or {}
dashboard\system_status.py:427:        llm_report = load_json(dd / "local_llm_report_latest.json", default={}) or {}
dashboard\system_status.py:584:        entries = load_jsonl_tail(dd / "signal_log.jsonl", max_entries=5)
dashboard\system_status.py:631:        ps = load_json(dd / "portfolio_state.json", default={}) or {}
dashboard\system_status.py:632:        pb = load_json(dd / "portfolio_state_bold.json", default={}) or {}
dashboard\system_status.py:765:        rows = load_jsonl_tail(
dashboard\system_status.py:777:    rows = load_jsonl(path)
scripts\backfill_accuracy_snapshots.py:373:    atomic_write_jsonl(jsonl_path, snaps)
scripts\backfill_accuracy_snapshots.py:450:            atomic_append_jsonl(args.output_jsonl, snap)
scripts\avanza_smoke_test.py:11:config = load_json("config.json")
scripts\fin_fish_monitor.py:437:            atomic_append_jsonl(LOG_PATH, {
scripts\fish_monitor_live.py:415:        atomic_append_jsonl(TRADE_LOG, entry)
scripts\fix_agent_dispatcher.py:205:    atomic_append_jsonl(CRITICAL_ERRORS_LOG, entry)
scripts\fish_preflight.py:86:    summary = load_json(SUMMARY_PATH) or {}
dashboard\app.py:108:    Switched from load_jsonl(limit=) (full scan + deque) to
dashboard\app.py:869:            decisions = load_jsonl_tail(decisions_path, max_entries=50)
portfolio\claude_gate.py:195:        atomic_append_jsonl(CRITICAL_ERRORS_LOG, entry)
portfolio\claude_gate.py:266:        atomic_append_jsonl(INVOCATIONS_LOG, entry)
portfolio\claude_gate.py:275:    for entry in load_jsonl(INVOCATIONS_LOG):
portfolio\claude_gate.py:658:    entries = load_jsonl(INVOCATIONS_LOG)
portfolio\alpha_vantage.py:39:    data = load_json(CACHE_FILE)
portfolio\alpha_vantage.py:55:        atomic_write_json(CACHE_FILE, snapshot)
portfolio\avanza_account_check.py:136:        atomic_append_jsonl(CRITICAL_ERRORS_LOG, entry)
portfolio\avanza_account_check.py:147:        cfg = load_json("config.json") or {}
portfolio\agent_invocation.py:156:    data = load_json(_STACK_OVERFLOW_FILE)
portfolio\agent_invocation.py:165:    atomic_write_json(_STACK_OVERFLOW_FILE, {
portfolio\agent_invocation.py:251:    atomic_append_jsonl(INVOCATIONS_FILE, entry)
portfolio\agent_invocation.py:269:    summary = load_json(summary_path, default=None)
portfolio\agent_invocation.py:293:        entries = load_jsonl(JOURNAL_FILE)
portfolio\agent_invocation.py:878:                (load_json(PATIENT_PORTFOLIO, default={}) or {}).get("transactions", [])
portfolio\agent_invocation.py:881:                (load_json(BOLD_PORTFOLIO, default={}) or {}).get("transactions", [])
portfolio\agent_invocation.py:904:            health = load_json(health_path, default={}) or {}
portfolio\agent_invocation.py:907:            atomic_write_json(health_path, health)
portfolio\agent_invocation.py:1033:        atomic_write_json(DATA_DIR / 'fishing_context.json', context)
portfolio\agent_invocation.py:1042:            atomic_write_json(DATA_DIR / 'fishing_context.json', {
portfolio\agent_invocation.py:1072:            state = load_json(pf_path, default={}) or {}
portfolio\agent_invocation.py:1249:        atomic_append_jsonl(INVOCATIONS_FILE, log_entry)
portfolio\agent_invocation.py:1363:    entries = load_jsonl(INVOCATIONS_FILE)
portfolio\accuracy_stats.py:106:    cache = load_json(SIGNAL_UTILITY_CACHE_FILE)
portfolio\accuracy_stats.py:134:            cache = load_json(SIGNAL_UTILITY_CACHE_FILE, default={})
portfolio\accuracy_stats.py:139:            _atomic_write_json(SIGNAL_UTILITY_CACHE_FILE, cache)
portfolio\accuracy_stats.py:163:    entries = load_jsonl_tail(SIGNAL_LOG, max_entries=50000)
portfolio\accuracy_stats.py:978:    cache = load_json(cache_file)
portfolio\accuracy_stats.py:988:            _atomic_write_json(cache_file, {"rates": rates, "time": time.time()})
portfolio\accuracy_stats.py:995:    cache = load_json(ACCURACY_CACHE_FILE)
portfolio\accuracy_stats.py:1012:        cache = load_json(ACCURACY_CACHE_FILE, default={})
portfolio\accuracy_stats.py:1020:        _atomic_write_json(ACCURACY_CACHE_FILE, cache)
portfolio\accuracy_stats.py:1127:    state = load_json(_DASHBOARD_PREWARM_STATE_FILE, default={}) or {}
portfolio\accuracy_stats.py:1137:        _atomic_write_json(
portfolio\accuracy_stats.py:1385:    cache = load_json(REGIME_ACCURACY_CACHE_FILE)
portfolio\accuracy_stats.py:1403:        cache = load_json(REGIME_ACCURACY_CACHE_FILE, default={})
portfolio\accuracy_stats.py:1408:        _atomic_write_json(REGIME_ACCURACY_CACHE_FILE, cache)
portfolio\accuracy_stats.py:1509:            cache = load_json(REGIME_ACCURACY_CACHE_FILE, default={})
portfolio\accuracy_stats.py:1515:                _atomic_write_json(REGIME_ACCURACY_CACHE_FILE, {})
portfolio\accuracy_stats.py:1522:                _atomic_write_json(REGIME_ACCURACY_CACHE_FILE, cache)
portfolio\accuracy_stats.py:1564:    atomic_append_jsonl(ACCURACY_SNAPSHOTS_FILE, snapshot)
portfolio\accuracy_stats.py:1582:    return load_jsonl(ACCURACY_SNAPSHOTS_FILE)
portfolio\accuracy_stats.py:1726:    cached = load_json(BEST_HORIZON_CACHE_FILE)
portfolio\accuracy_stats.py:1800:            _atomic_write_json(BEST_HORIZON_CACHE_FILE, {"time": time.time(), "data": result})
portfolio\accuracy_stats.py:1920:    cache = load_json(TICKER_ACCURACY_CACHE_FILE)
portfolio\accuracy_stats.py:1938:        cache = load_json(TICKER_ACCURACY_CACHE_FILE, default={})
portfolio\accuracy_stats.py:1943:        _atomic_write_json(TICKER_ACCURACY_CACHE_FILE, cache)
portfolio\avanza_orders.py:70:    result = load_json(PENDING_FILE, default=[])
portfolio\avanza_orders.py:79:    atomic_write_json(PENDING_FILE, orders)
portfolio\avanza_orders.py:260:    offset_data = load_json(offset_file)
portfolio\avanza_orders.py:341:        atomic_write_json(offset_file, {"offset": offset})
portfolio\accuracy_degradation.py:91:    state = load_json(ALERT_STATE_FILE, default={})
portfolio\accuracy_degradation.py:102:    atomic_write_json(ALERT_STATE_FILE, state)
portfolio\accuracy_degradation.py:106:    state = load_json(SNAPSHOT_STATE_FILE, default={})
portfolio\accuracy_degradation.py:114:    atomic_write_json(SNAPSHOT_STATE_FILE, state)
portfolio\avanza_client.py:50:    config = load_json(CONFIG_FILE)
portfolio\avanza_tracker.py:35:    config = load_json(CONFIG_FILE, default={})
portfolio\analyze.py:64:    all_entries = load_jsonl(JOURNAL_FILE)
portfolio\analyze.py:81:            pf = load_json(filepath)
portfolio\analyze.py:233:        atomic_append_jsonl(ANALYSIS_LOG_FILE, {
portfolio\analyze.py:264:    summary = load_json(AGENT_SUMMARY_FILE)
portfolio\analyze.py:315:            config = load_json(CONFIG_FILE)
portfolio\analyze.py:373:    return load_json(AGENT_SUMMARY_FILE)
portfolio\analyze.py:405:        atomic_append_jsonl(WATCH_LOG_FILE, event)
portfolio\analyze.py:618:    config = load_json(CONFIG_FILE)
portfolio\autonomous.py:58:        summary = load_json(path, default=None)
portfolio\autonomous.py:100:    prev_entries = load_jsonl(JOURNAL_FILE, limit=5)
portfolio\autonomous.py:151:    atomic_append_jsonl(JOURNAL_FILE, journal_entry)
portfolio\autonomous.py:166:    atomic_append_jsonl(DECISIONS_FILE, decision_log)
portfolio\autonomous.py:670:    compact = load_json(compact_file)
portfolio\autonomous.py:808:    data = load_json(THROTTLE_FILE, default={})
portfolio\autonomous.py:826:        atomic_write_json(THROTTLE_FILE, data)
scripts\health_check.py:127:    trigger = load_json(DATA_DIR / "trigger_state.json", {})
scripts\health_check.py:144:    hs = load_json(DATA_DIR / "health_state.json", {})
scripts\health_check.py:193:    summary = load_json(DATA_DIR / "agent_summary.json", {})
scripts\health_check.py:279:    hs = load_json(DATA_DIR / "health_state.json", {})
scripts\health_check.py:288:    entries = load_jsonl_tail(DATA_DIR / "telegram_messages.jsonl", max_entries=20)
scripts\health_check.py:300:    digest_state = load_json(DATA_DIR / "digest_state.json", {})
scripts\health_check.py:305:    daily_state = load_json(DATA_DIR / "daily_digest_state.json", {})
scripts\health_check.py:307:        load_json(DATA_DIR / "trigger_state.json", {}).get("last_daily_digest_time")
scripts\health_check.py:321:    if load_json(DATA_DIR / "portfolio_state.json") is None:
scripts\health_check.py:328:    if load_json(DATA_DIR / "prophecy.json") is None:
scripts\health_check.py:481:        hs = load_json(DATA_DIR / "health_state.json", {})
scripts\iskbet.py:83:def atomic_write_json(path, data):
scripts\iskbet.py:194:        atomic_write_json(ISKBETS_CONFIG, cfg)
scripts\iskbet.py:226:    atomic_write_json(ISKBETS_CONFIG, cfg)
portfolio\crypto_scheduler.py:49:    state = load_json(STATE_FILE, default={})
portfolio\crypto_scheduler.py:55:    atomic_write_json(STATE_FILE, state)
portfolio\crypto_scheduler.py:110:    summary = load_json(DATA_DIR / "agent_summary_compact.json")
portfolio\crypto_scheduler.py:277:        fund_cache = load_json(DATA_DIR / "fundamentals_cache.json", default={})
portfolio\crypto_scheduler.py:371:            atomic_append_jsonl(LOG_FILE, log_entry)
portfolio\file_utils.py:45:def atomic_write_json(path, data, indent=2, ensure_ascii=True):
portfolio\file_utils.py:66:def load_json(path, default=None):
portfolio\file_utils.py:89:def require_json(path):
portfolio\file_utils.py:92:    Unlike load_json(), this function does NOT silently return defaults.
portfolio\file_utils.py:104:def load_jsonl(path, limit=None):
portfolio\file_utils.py:136:def load_jsonl_tail(path, max_entries=500, tail_bytes=512_000):
portfolio\file_utils.py:139:    Much more efficient than load_jsonl(limit=N) for large files because
portfolio\file_utils.py:203:def jsonl_sidecar_lock(path):
portfolio\file_utils.py:261:def atomic_append_jsonl(path, entry):
portfolio\file_utils.py:280:    with jsonl_sidecar_lock(path):
portfolio\file_utils.py:287:def atomic_write_jsonl(path, entries):
portfolio\file_utils.py:349:def prune_jsonl(path, max_entries=5000):
portfolio\bigbet.py:37:    state = load_json(STATE_FILE)
portfolio\bigbet.py:44:    atomic_write_json(STATE_FILE, state)
portfolio\bigbet.py:87:        summary = load_json(summary_file)
portfolio\bigbet.py:212:        atomic_append_jsonl(EVAL_LOG_FILE, {
portfolio\equity_curve.py:50:    result = load_jsonl(path)
portfolio\crypto_precompute.py:70:    state = load_json(_STATE_FILE, default={})
portfolio\crypto_precompute.py:79:        atomic_write_json(_STATE_FILE, {
portfolio\crypto_precompute.py:88:        atomic_write_json(_STATE_FILE, {
portfolio\crypto_precompute.py:102:    atomic_write_json(_OUTPUT_FILE, ctx)
portfolio\crypto_macro_data.py:210:        summary = load_json(DATA_DIR / "agent_summary_compact.json")
portfolio\crypto_macro_data.py:308:        atomic_append_jsonl(RATIO_HISTORY_FILE, entry)
portfolio\crypto_macro_data.py:424:        atomic_append_jsonl(NETFLOW_HISTORY_FILE, entry)
scripts\loop_health_report.py:189:        cfg = load_json("config.json") or {}
portfolio\fear_greed.py:26:        data = load_json(_STREAK_FILE)
portfolio\fear_greed.py:43:    data = load_json(_STREAK_FILE, default={}) or {}
portfolio\fear_greed.py:77:        atomic_write_json(_STREAK_FILE, data)
portfolio\config_validator.py:66:    config = load_json(CONFIG_FILE)
scripts\loop_health_watchdog.py:41:    return load_json(str(STATE_FILE)) or {"last_alert_per_loop": {}}
scripts\loop_health_watchdog.py:46:        atomic_write_json(str(STATE_FILE), state)
scripts\loop_health_watchdog.py:123:        cfg = load_json(str(REPO / "config.json")) or {}
portfolio\cumulative_tracker.py:52:    atomic_append_jsonl(SNAPSHOTS_FILE, entry)
portfolio\cumulative_tracker.py:105:        snapshots = load_jsonl(SNAPSHOTS_FILE)
portfolio\daily_digest.py:32:    state = load_json(_DAILY_DIGEST_STATE_FILE, default={})
portfolio\daily_digest.py:35:        old = load_json(DATA_DIR / "trigger_state.json", default={})
portfolio\daily_digest.py:43:    state = load_json(_DAILY_DIGEST_STATE_FILE, default={})
portfolio\daily_digest.py:47:    atomic_write_json(_DAILY_DIGEST_STATE_FILE, state)
portfolio\daily_digest.py:106:    summary = load_json(AGENT_SUMMARY_FILE, default={})
portfolio\daily_digest.py:228:    bold = load_json(BOLD_STATE_FILE)
portfolio\decision_outcome_tracker.py:29:    decisions = load_jsonl(DECISIONS_FILE)
portfolio\decision_outcome_tracker.py:35:    existing_outcomes = load_jsonl(OUTCOMES_FILE)
portfolio\decision_outcome_tracker.py:99:                atomic_append_jsonl(OUTCOMES_FILE, outcome)
portfolio\forecast_signal.py:339:        summary = load_json(AGENT_SUMMARY_FILE)
portfolio\forecast_signal.py:384:            atomic_append_jsonl(PREDICTIONS_FILE, entry)
portfolio\digest.py:37:        state = load_json(_DIGEST_STATE_FILE, default={})
portfolio\digest.py:40:            old = load_json(DATA_DIR / "trigger_state.json", default={})
portfolio\digest.py:48:    state = load_json(_DIGEST_STATE_FILE, default={})
portfolio\digest.py:52:    _atomic_write_json(_DIGEST_STATE_FILE, state)
portfolio\digest.py:66:    entries = load_jsonl_tail(INVOCATIONS_FILE, max_entries=500)
portfolio\digest.py:100:    journal = load_jsonl_tail(JOURNAL_FILE, max_entries=500)
portfolio\digest.py:122:    signal_entries = load_jsonl_tail(SIGNAL_LOG_FILE, max_entries=500)
portfolio\digest.py:140:    summary = load_json(AGENT_SUMMARY_FILE, default={})
portfolio\digest.py:219:            bold = load_json(BOLD_STATE_FILE, default={})
portfolio\fin_snipe_manager.py:102:        config = load_json(BASE_DIR / "config.json", default={})
portfolio\fin_snipe_manager.py:151:    atomic_append_jsonl(path, entry)
portfolio\fin_snipe_manager.py:204:            prune_jsonl(path, max_entries=LOG_MAX_ENTRIES)
portfolio\fin_snipe_manager.py:755:        atomic_append_jsonl(prediction_log_path, prediction_entry)
portfolio\fin_snipe_manager.py:889:    state = load_json(path, default=_default_state())
portfolio\fin_snipe_manager.py:899:    atomic_write_json(path, state, ensure_ascii=False)
portfolio\fin_snipe.py:32:def _load_json(path: Path) -> dict:
portfolio\fin_snipe.py:34:    return load_json(path) or {}
portfolio\fin_snipe.py:84:        analysis = _load_json(SILVER_ANALYSIS_PATH)
portfolio\fin_snipe.py:102:    summary = _load_json(SUMMARY_PATH)
portfolio\forecast_accuracy.py:71:    return load_jsonl(str(path))
portfolio\forecast_accuracy.py:78:    for entry in load_jsonl(str(path)):
portfolio\forecast_accuracy.py:357:    for snap in load_jsonl(str(path)):
portfolio\forecast_accuracy.py:375:    atomic_write_jsonl(path, entries)
portfolio\fin_fish.py:287:    summary = load_json(SUMMARY_PATH) or {}
portfolio\fin_fish.py:1187:        signal_data = load_json(BASE_DIR / "data" / "agent_summary_compact.json")
portfolio\fin_fish.py:1221:                deep_ctx = load_json(precompute_path)
portfolio\fin_fish.py:1348:    atomic_append_jsonl(FISH_LOG_PATH, log_entry)
portfolio\fin_fish.py:1392:                pos_state = load_json(BASE_DIR / "data" / "metals_positions_state.json") or {}
portfolio\elongir\runner.py:58:            data = load_json(config_path, default=None)
portfolio\elongir\state.py:122:        atomic_write_json(path, data)
portfolio\elongir\state.py:127:        data = load_json(path)
portfolio\elongir\state.py:173:    atomic_append_jsonl(trades_file, entry)
portfolio\elongir\state.py:205:    atomic_append_jsonl(log_file, entry)
portfolio\fin_evolve.py:55:def _atomic_append_jsonl(path, entries):
portfolio\fin_evolve.py:58:        atomic_append_jsonl(path, entry)
portfolio\fin_evolve.py:86:    entries = load_jsonl(_PRICE_FILE)
portfolio\fin_evolve.py:205:    entries = load_jsonl(_LOG_FILE)
portfolio\fin_evolve.py:251:        atomic_write_jsonl(_LOG_FILE, entries)
portfolio\fin_evolve.py:321:    journal = load_jsonl(_JOURNAL_FILE)
portfolio\fin_evolve.py:337:    existing = load_jsonl(_JOURNAL_OUTCOMES_FILE)
portfolio\fin_evolve.py:418:        atomic_write_jsonl(_JOURNAL_OUTCOMES_FILE, scored_existing + new_outcomes)
portfolio\fin_evolve.py:425:        _atomic_append_jsonl(_JOURNAL_OUTCOMES_FILE, new_outcomes)
portfolio\fin_evolve.py:905:    fin_entries = load_jsonl(_LOG_FILE)
portfolio\fin_evolve.py:909:    journal_outcomes = load_jsonl(_JOURNAL_OUTCOMES_FILE)
portfolio\fin_evolve.py:1039:    atomic_write_json(_LESSONS_FILE, lessons)
portfolio\fin_evolve.py:1042:    atomic_write_json(_LEGACY_LESSONS_FILE, lessons)
portfolio\fin_evolve.py:1136:    state = load_json(_EVOLVE_STATE_FILE, default={})
portfolio\fin_evolve.py:1147:        atomic_write_json(
scripts\oil_loop_scorecard.py:41:def load_jsonl(path: Path) -> list[dict[str, Any]]:
scripts\oil_loop_scorecard.py:206:    decisions = load_jsonl(DECISIONS_LOG)
scripts\oil_loop_scorecard.py:207:    trades = load_jsonl(TRADES_LOG)
portfolio\fish_monitor_smart.py:170:        summary = load_json(SUMMARY_PATH) or {}
portfolio\fish_monitor_smart.py:205:        full_summary = load_json(BASE_DIR / "data" / "agent_summary.json") or {}
portfolio\fish_monitor_smart.py:710:                atomic_append_jsonl(MONITOR_LOG, log_entry)
portfolio\fish_monitor_smart.py:737:        atomic_write_json(MONITOR_STATE, final_state)
portfolio\focus_analysis.py:86:    entries = load_jsonl(JOURNAL_FILE, limit=400)
portfolio\focus_analysis.py:154:    summary = load_json(SUMMARY_FILE)
portfolio\focus_analysis.py:157:    config = load_json(CONFIG_FILE, default={})
portfolio\gold_precompute.py:33:    config = load_json("config.json")
portfolio\health.py:41:        atomic_write_json(HEALTH_FILE, state)
portfolio\health.py:46:    state = load_json(HEALTH_FILE)
portfolio\health.py:61:        atomic_write_json(HEALTH_FILE, state)
portfolio\health.py:86:        atomic_write_json(HEALTH_FILE, state)
portfolio\health.py:196:            atomic_write_json(HEALTH_FILE, wb_state)
portfolio\health.py:247:            atomic_write_json(HEALTH_FILE, state)
portfolio\health.py:251:            atomic_write_json(HEALTH_FILE, state)
portfolio\health.py:298:        atomic_write_json(HEALTH_FILE, state)
portfolio\health.py:383:    entries = load_jsonl_tail(signal_log, max_entries=50)
portfolio\health.py:432:    entries = load_jsonl_tail(signal_log, max_entries=recent_entries)
portfolio\grid_fisher.py:337:    raw = load_json(state_path, default=None)
portfolio\grid_fisher.py:367:        atomic_write_json(state_path, state.to_dict())
portfolio\grid_fisher.py:423:    atomic_append_jsonl(decisions_path, entry)
scripts\probe_oil_warrants.py:68:            cached = load_json(str(REPO / CATALOG_FILE)) or {}
scripts\probe_oil_warrants.py:80:            atomic_write_json(str(REPO / CATALOG_FILE), payload)
scripts\replay_consensus.py:366:        atomic_write_json(str(out_path), summary)
portfolio\ic_computation.py:247:    atomic_write_json(IC_CACHE_FILE, cache)
portfolio\ic_computation.py:255:    cache = load_json(IC_CACHE_FILE)
scripts\pf.py:55:def load_json(path):
scripts\pf.py:63:def _atomic_write_json(path, data):
scripts\pf.py:162:    state = load_json(STATE_FILE)
scripts\pf.py:163:    summary = load_json(SUMMARY_FILE)
scripts\pf.py:208:    summary = load_json(SUMMARY_FILE)
scripts\pf.py:264:    state = load_json(STATE_FILE)
scripts\pf.py:308:        trigger = load_json(TRIGGER_FILE)
scripts\pf.py:324:    state = load_json(STATE_FILE)
scripts\pf.py:358:    summary = load_json(SUMMARY_FILE)
scripts\pf.py:359:    state = load_json(STATE_FILE)
scripts\pf.py:415:    _atomic_write_json(STATE_FILE, state)
scripts\pf.py:436:    summary = load_json(SUMMARY_FILE)
scripts\pf.py:437:    state = load_json(STATE_FILE)
scripts\pf.py:482:    _atomic_write_json(STATE_FILE, state)
portfolio\golddigger\state.py:47:        atomic_write_json(path, data)
portfolio\golddigger\state.py:52:        data = load_json(path)
portfolio\golddigger\state.py:133:    atomic_append_jsonl(trades_file, entry)
portfolio\golddigger\state.py:183:    atomic_append_jsonl(log_file, entry)
portfolio\journal.py:184:        pf = load_json(filepath)
portfolio\journal.py:436:    return load_json(config_file, default={}) or {}
portfolio\journal.py:447:        summary = load_json(summary_file)
portfolio\journal.py:455:            pf = load_json(DATA_DIR / fname)
portfolio\golddigger\runner.py:81:            data = load_json(config_path, default=None)
portfolio\iskbets.py:40:    cfg = load_json(CONFIG_FILE)
portfolio\iskbets.py:63:    atomic_write_json(CONFIG_FILE, cfg)
portfolio\iskbets.py:68:    result = load_json(STATE_FILE)
portfolio\iskbets.py:76:    atomic_write_json(STATE_FILE, state)
portfolio\iskbets.py:256:        summary = load_json(summary_file)
portfolio\iskbets.py:357:        atomic_append_jsonl(GATE_LOG_FILE, {
portfolio\golddigger\data_provider.py:416:    return load_json(path, default=None)
scripts\verify_tunnel.py:46:    cfg = load_json(CONFIG, default={}) or {}
portfolio\linear_factor.py:176:        atomic_write_json(path, data)
portfolio\linear_factor.py:184:        data = load_json(path)
portfolio\kelly_metals.py:51:    cache = load_json(str(ACCURACY_CACHE), default={})
portfolio\llm_probability_log.py:159:        atomic_append_jsonl(log_path or _PROB_LOG, entry)
scripts\verify_tunnel_alerted.py:76:    atomic_append_jsonl(
scripts\verify_tunnel_alerted.py:92:    cfg = load_json(PROJECT / "config.json", default={}) or {}
portfolio\llm_outcome_backfill.py:250:            atomic_append_jsonl(outcomes_path, outcome_row)
portfolio\local_llm_report.py:181:        result = load_json(CONFIG_EXAMPLE_FILE)
portfolio\local_llm_report.py:383:    atomic_write_json(latest_path, entry)
portfolio\local_llm_report.py:384:    atomic_append_jsonl(history_path, entry)
portfolio\local_llm_report.py:385:    prune_jsonl(history_path, max_entries=max_entries)
portfolio\local_llm_report.py:386:    atomic_write_json(
portfolio\local_llm_report.py:415:    state = load_json(state_path, default={}) or {}
portfolio\kelly_sizing.py:260:    portfolio = load_json(portfolio_path, default={})
portfolio\kelly_sizing.py:266:        agent_summary = load_json(AGENT_SUMMARY_FILE, default={})
portfolio\kelly_sizing.py:350:    agent_summary = load_json(AGENT_SUMMARY_FILE, default={})
portfolio\loop_contract.py:292:    cfg = load_json(CONFIG_FILE, default={}) or {}
portfolio\loop_contract.py:297:    health = load_json(HEALTH_STATE_FILE)
portfolio\loop_contract.py:411:    contract_state = load_json(CONTRACT_STATE_FILE, default={}) or {}
portfolio\loop_contract.py:509:        atomic_write_json(CONTRACT_STATE_FILE, contract_state)
portfolio\loop_contract.py:1034:        state = load_json(state_path, default=None)
portfolio\loop_contract.py:1431:        state = load_json(state_file, default={}) or {}
portfolio\loop_contract.py:1503:        existing = load_json(state_file, default={}) or {}
portfolio\loop_contract.py:1507:        atomic_write_json(state_file, existing)
portfolio\loop_contract.py:1740:        state = load_json(state_file, default={})
portfolio\loop_contract.py:1815:        existing = load_json(self._state_file, default={}) or {}
portfolio\loop_contract.py:1819:        atomic_write_json(self._state_file, existing)
portfolio\loop_contract.py:1869:        atomic_append_jsonl(CONTRACT_LOG_FILE, {
portfolio\loop_contract.py:2067:    state = load_json(state_file, default={}) or {}
portfolio\loop_contract.py:2122:        existing = load_json(state_file, default={}) or {}
portfolio\loop_contract.py:2137:        atomic_write_json(state_file, existing)
portfolio\loop_contract.py:2162:        existing = load_json(state_file, default={}) or {}
portfolio\loop_contract.py:2166:        atomic_write_json(state_file, existing)
portfolio\loop_contract.py:2336:        _state_pre = load_json(tracker_state_file, default={}) or {}
portfolio\llm_prewarmer.py:173:        atomic_append_jsonl(STATE_FILE, entry)
portfolio\loop_health.py:221:        atomic_write_json(str(path), payload)
portfolio\log_rotation.py:270:    with jsonl_sidecar_lock(filepath):
portfolio\main.py:362:            prune_jsonl(DATA_DIR / name, max_entries=5000)
portfolio\main.py:977:    data = load_json(_CRASH_COUNTER_FILE)
portfolio\main.py:985:    atomic_write_json(_CRASH_COUNTER_FILE, {
portfolio\main.py:1015:                config = load_json(config_path, default={})
portfolio\main.py:1028:        config = load_json(config_path, default={})
portfolio\main.py:1362:        _summary = load_json(_data / "agent_summary.json", default={})
portfolio\main.py:1363:        _patient = load_json(_data / "portfolio_state.json", default={})
portfolio\main.py:1364:        _bold = load_json(_data / "portfolio_state_bold.json", default={})
portfolio\market_health.py:405:    prev_ftd = load_json(_STATE_FILE, default={}).get("ftd_state")
portfolio\market_health.py:454:    atomic_write_json(_STATE_FILE, state_to_save)
portfolio\memory_consolidation.py:43:    entries = load_jsonl_tail(path, max_entries=50_000, tail_bytes=50_000_000)
portfolio\message_store.py:102:    atomic_append_jsonl(MESSAGES_FILE, entry)
portfolio\message_throttle.py:39:    state = load_json(PENDING_FILE, default={})
portfolio\message_throttle.py:61:    state = load_json(PENDING_FILE, default={})
portfolio\message_throttle.py:64:    atomic_write_json(PENDING_FILE, state)
portfolio\message_throttle.py:83:    state = load_json(PENDING_FILE, default={})
portfolio\message_throttle.py:107:    state = load_json(PENDING_FILE, default={})
portfolio\message_throttle.py:111:    atomic_write_json(PENDING_FILE, state)
portfolio\metals_precompute.py:63:    state = load_json(_STATE_FILE, default={})
portfolio\metals_precompute.py:72:        atomic_write_json(_STATE_FILE, {
portfolio\metals_precompute.py:81:        atomic_write_json(_STATE_FILE, {
portfolio\metals_precompute.py:108:    atomic_write_json("data/silver_deep_context.json", silver_ctx)
portfolio\metals_precompute.py:115:    atomic_write_json("data/gold_deep_context.json", gold_ctx)
portfolio\metals_precompute.py:133:    refresh_state = load_json(_REFRESH_STATE_FILE, default={})
portfolio\metals_precompute.py:243:        atomic_write_json(_REFRESH_STATE_FILE, refresh_state)
portfolio\metals_precompute.py:589:    existing = load_jsonl(_COT_HISTORY_FILE)
portfolio\metals_precompute.py:598:            prune_jsonl(_COT_HISTORY_FILE, max_entries=104)
portfolio\metals_precompute.py:612:    atomic_append_jsonl(_COT_HISTORY_FILE, record)
portfolio\metals_precompute.py:618:    entries = load_jsonl(_COT_HISTORY_FILE)
portfolio\metals_precompute.py:806:    summary = load_json("data/agent_summary_compact.json")
portfolio\metals_precompute.py:912:    entries = load_jsonl("data/layer2_journal.jsonl")
portfolio\metals_precompute.py:939:    entries = load_jsonl("data/metals_signal_log.jsonl")
portfolio\metals_precompute.py:962:    prophecy = load_json("data/prophecy.json")
portfolio\metals_precompute.py:1060:    existing = load_json("data/silver_external_cache.json")
portfolio\metals_precompute.py:1148:    existing = load_json("data/gold_external_cache.json")
portfolio\metals_precompute.py:1217:    config = load_json("config.json")
portfolio\meta_learner.py:344:    atomic_write_json(metrics_path, metrics)
portfolio\meta_learner.py:395:        _m = load_json(metrics_path, default={})
portfolio\meta_learner.py:443:        m = load_json(metrics_path, default={})
portfolio\microstructure_state.py:213:        atomic_write_json(_STATE_FILE, state)
portfolio\microstructure_state.py:222:    data = load_json(_STATE_FILE)
portfolio\mstr_loop\telegram_report.py:44:        atomic_write_json(path, state, ensure_ascii=False)
portfolio\mstr_loop\loop.py:66:        atomic_append_jsonl(config.POLL_LOG, record)
portfolio\mstr_loop\execution.py:554:        atomic_append_jsonl(config.SHADOW_LOG, record)
portfolio\mstr_loop\execution.py:579:        atomic_append_jsonl(config.TRADES_LOG, record)
portfolio\mstr_loop\data_provider.py:156:        summary = load_json(agent_summary_path)
portfolio\avanza_session.py:79:    data = load_json(SESSION_FILE)
portfolio\avanza_session.py:109:        data = load_json(SESSION_FILE)
portfolio\mstr_loop\state.py:186:        raw = load_json(path)
portfolio\mstr_loop\state.py:199:        atomic_write_json(path, _state_to_dict(state), ensure_ascii=False)
portfolio\mstr_precompute.py:55:    state = load_json(_STATE_FILE, default={})
portfolio\mstr_precompute.py:63:        atomic_write_json(_STATE_FILE, {
portfolio\mstr_precompute.py:72:        atomic_write_json(_STATE_FILE, {
portfolio\mstr_precompute.py:114:    atomic_write_json(_OUTPUT_FILE, ctx)
portfolio\oil_grid_signal.py:158:    cached = load_json(SIGNAL_FILE, default=None)
portfolio\oil_grid_signal.py:173:        atomic_write_json(SIGNAL_FILE, fresh)
portfolio\oil_precompute.py:65:    state = load_json(_STATE_FILE, default={})
portfolio\oil_precompute.py:74:        atomic_write_json(_STATE_FILE, {
portfolio\oil_precompute.py:83:        atomic_write_json(_STATE_FILE, {
portfolio\oil_precompute.py:103:    atomic_write_json(_OUTPUT_FILE, ctx)
portfolio\oil_precompute.py:117:    refresh_state = load_json(_REFRESH_STATE_FILE, default={})
portfolio\oil_precompute.py:227:        atomic_write_json(_REFRESH_STATE_FILE, refresh_state)
portfolio\oil_precompute.py:681:    existing = load_jsonl(_COT_HISTORY_FILE)
portfolio\oil_precompute.py:689:            prune_jsonl(_COT_HISTORY_FILE, max_entries=52)
portfolio\oil_precompute.py:703:    atomic_append_jsonl(_COT_HISTORY_FILE, record)
portfolio\oil_precompute.py:709:    entries = load_jsonl(_COT_HISTORY_FILE)
portfolio\oil_precompute.py:995:    existing = load_json("data/oil_external_cache.json")
portfolio\oil_precompute.py:1083:    config = load_json("config.json")
portfolio\onchain_data.py:100:        atomic_write_json(CACHE_FILE, data, ensure_ascii=False)
portfolio\onchain_data.py:107:    data = load_json(CACHE_FILE)
portfolio\onchain_data.py:252:    persistent = load_json(CACHE_FILE, default={})
portfolio\orb_postmortem.py:135:    atomic_append_jsonl(filepath, entry)
portfolio\outcome_tracker.py:157:    atomic_append_jsonl(SIGNAL_LOG, entry)
portfolio\perception_gate.py:91:    result = load_json(path)
portfolio\perception_gate.py:95:    return load_json(path)
portfolio\regime_alerts.py:39:    entries = load_jsonl(REGIME_HISTORY_FILE)
portfolio\regime_alerts.py:97:    atomic_append_jsonl(REGIME_HISTORY_FILE, entry)
portfolio\regime_alerts.py:111:    entries = load_jsonl(REGIME_HISTORY_FILE)
portfolio\regime_alerts.py:144:    entries = load_jsonl(REGIME_HISTORY_FILE)
portfolio\regime_alerts.py:176:    config = load_json(CONFIG_FILE, default={})
portfolio\reflection.py:93:    entries = load_jsonl(JOURNAL_FILE, limit=100)
portfolio\reflection.py:149:    patient = load_json(PORTFOLIO_FILE, {})
portfolio\reflection.py:150:    bold = load_json(BOLD_FILE, {})
portfolio\reflection.py:154:    reflections = load_jsonl(REFLECTIONS_FILE)
portfolio\reflection.py:182:    patient = load_json(PORTFOLIO_FILE, {})
portfolio\reflection.py:183:    bold = load_json(BOLD_FILE, {})
portfolio\reflection.py:204:    atomic_append_jsonl(REFLECTIONS_FILE, reflection)
portfolio\reflection.py:228:    reflections = load_jsonl(REFLECTIONS_FILE)
portfolio\prophecy.py:61:    data = load_json(PROPHECY_FILE)
portfolio\prophecy.py:73:    atomic_write_json(PROPHECY_FILE, data)
portfolio\portfolio_validator.py:263:    # but malformed" portfolio. load_json() retries on transient JSON
portfolio\portfolio_validator.py:268:    portfolio = load_json(p)
portfolio\portfolio_mgr.py:84:    loaded = load_json(str(path), default=None)
portfolio\portfolio_mgr.py:97:                loaded = load_json(str(bak), default=None)
portfolio\portfolio_mgr.py:113:        _atomic_write_json(path, state)
portfolio\portfolio_mgr.py:158:        _atomic_write_json(path, state)
portfolio\shadow_registry.py:68:    data = load_json(str(p), default=None)
portfolio\shadow_registry.py:78:    atomic_write_json(str(p), data)
portfolio\sentiment.py:739:        atomic_append_jsonl(AB_LOG_FILE, entry)
portfolio\sentiment_shadow_backfill.py:285:                atomic_append_jsonl(outcomes_path, outcome_row)
portfolio\seasonality.py:121:    atomic_write_json(_STATE_FILE, profiles)
portfolio\seasonality.py:126:    return load_json(_STATE_FILE) or {}
portfolio\signal_decay_alert.py:146:        atomic_append_jsonl("data/signal_decay_alerts.jsonl", entry)
portfolio\risk_management.py:156:            atomic_write_json(DATA_DIR / _FX_CACHE_FILENAME, {
portfolio\risk_management.py:165:    cached = load_json(DATA_DIR / _FX_CACHE_FILENAME, default=None)
portfolio\risk_management.py:239:    portfolio = load_json(portfolio_path, default={})
portfolio\risk_management.py:249:        summary = load_json(agent_summary_path, default={})
portfolio\risk_management.py:590:    patient = load_json(patient_path, default={})
portfolio\risk_management.py:591:    bold = load_json(bold_path, default={})
portfolio\risk_management.py:596:    summary = load_json(agent_summary_path, default={"signals": {}})
portfolio\risk_management.py:625:    atomic_append_jsonl(history_path, entry)
portfolio\signal_history.py:39:    return load_jsonl(HISTORY_FILE)
portfolio\signal_history.py:48:    atomic_write_jsonl(HISTORY_FILE, entries)
portfolio\reporting.py:455:    _patient_pf = load_json(DATA_DIR / "portfolio_state.json", default={})
portfolio\reporting.py:456:    _bold_pf = load_json(DATA_DIR / "portfolio_state_bold.json", default={})
portfolio\reporting.py:687:        warrant_state = load_json(warrant_state_path)
portfolio\reporting.py:748:    prev = load_json(AGENT_SUMMARY_FILE)
portfolio\reporting.py:808:    atomic_write_json(AGENT_SUMMARY_FILE, summary)
portfolio\reporting.py:837:        prev = load_json(SIGNAL_STATE_SINCE_FILE, default={})
portfolio\reporting.py:840:        atomic_write_json(SIGNAL_STATE_SINCE_FILE, payload)
portfolio\reporting.py:859:        state = load_json(DATA_DIR / fname, default={})
portfolio\reporting.py:986:        system_lessons = load_json(
portfolio\reporting.py:1018:    atomic_write_json(COMPACT_SUMMARY_FILE, compact)
portfolio\reporting.py:1048:        state = load_json(state_file, default={})
portfolio\reporting.py:1201:    atomic_write_json(TIER1_FILE, t1)
portfolio\reporting.py:1314:    atomic_write_json(TIER2_FILE, t2)
portfolio\signal_engine.py:1020:            data = _load_json(str(_SENTIMENT_STATE_FILE), default=None)
portfolio\signal_engine.py:1063:        atomic_write_json(_SENTIMENT_STATE_FILE, {"prev_sentiment": snapshot})
portfolio\signal_engine.py:1351:        cache = load_json(DATA_DIR / "accuracy_cache.json")
portfolio\signal_postmortem.py:243:    atomic_write_json(POSTMORTEM_FILE, report)
portfolio\signal_postmortem.py:257:    data = load_json(POSTMORTEM_FILE)
portfolio\signals\claude_fundamental.py:671:            atomic_append_jsonl(_CF_LOG, entry)
portfolio\signals\claude_fundamental.py:687:    summary = load_json(summary_path, default={})
portfolio\signals\claude_fundamental.py:706:                portfolios[pf] = load_json(pf_path, default={})
portfolio\signals\claude_fundamental.py:765:        entries = load_jsonl_tail(_CF_LOG, max_entries=400)
portfolio\signals\claude_fundamental.py:811:        entries = load_jsonl_tail(_CF_LOG, max_entries=500)
portfolio\signals\credit_spread.py:285:            cfg = load_json("config.json", default={}) or {}
portfolio\signals\cot_positioning.py:68:    ctx = load_json(path, default=None)
portfolio\signals\cot_positioning.py:83:    entries = load_jsonl(str(_DATA_DIR / "cot_history.jsonl"))
portfolio\silver_precompute.py:33:    config = load_json("config.json")
portfolio\stats.py:13:    entries = load_jsonl(INVOCATIONS_FILE)
portfolio\stats.py:50:    entries = load_jsonl(TELEGRAMS_FILE)
portfolio\stats.py:93:    invocations = load_jsonl(INVOCATIONS_FILE)
portfolio\stats.py:94:    telegrams = load_jsonl(TELEGRAMS_FILE)
portfolio\signal_weight_optimizer.py:161:    atomic_write_json(path, result.to_dict())
portfolio\signal_weight_optimizer.py:167:    data = load_json(path)
portfolio\signal_weights.py:103:        atomic_write_json(self._path, payload)
portfolio\signal_weights.py:111:        data = load_json(self._path, default=None)
portfolio\signals\forecast.py:89:        _cfg = _load_json(
portfolio\signals\forecast.py:232:        atomic_append_jsonl(_HEALTH_FILE, entry)
portfolio\signals\forecast.py:340:            cfg = load_json(str(Path(__file__).resolve().parent.parent.parent / "config.json"), {})
portfolio\signals\forecast.py:964:            atomic_append_jsonl(_PREDICTIONS_FILE, entry)
portfolio\telegram_notifications.py:130:    bold = load_json(BOLD_STATE_FILE)
portfolio\trade_guards.py:37:    return load_json(str(STATE_FILE), default={
portfolio\trade_guards.py:47:    atomic_write_json(STATE_FILE, state)
portfolio\trade_guards.py:66:        pf = load_json(str(DATA_DIR / pf_name), default={})
portfolio\signals\gold_real_yield_paradox.py:265:            cfg = load_json("config.json")
portfolio\telegram_poller.py:74:            state = load_json(POLLER_STATE_FILE, default=None)
portfolio\telegram_poller.py:98:            atomic_write_json(
portfolio\telegram_poller.py:289:            atomic_append_jsonl(INBOUND_LOG, entry)
portfolio\telegram_poller.py:345:        cfg = load_json(config_path, default={})
portfolio\telegram_poller.py:361:        atomic_write_json(config_path, cfg)
portfolio\trigger.py:111:    return load_json(STATE_FILE, default={})
portfolio\trigger.py:126:    atomic_write_json(STATE_FILE, state)
portfolio\trigger.py:141:            pf = load_json(pf_file, default=None)
portfolio\train_signal_weights.py:54:    entries = load_jsonl(log_path)
portfolio\weekly_digest.py:27:def _load_jsonl(path, since=None):
portfolio\weekly_digest.py:29:    entries = load_jsonl(path)
portfolio\weekly_digest.py:138:    patient_state = load_json(PATIENT_FILE, default={})
portfolio\weekly_digest.py:139:    bold_state = load_json(BOLD_FILE, default={})
portfolio\weekly_digest.py:154:    signal_entries = _load_jsonl(SIGNAL_LOG, since=week_ago)
portfolio\weekly_digest.py:168:    journal_entries = _load_jsonl(JOURNAL_FILE, since=week_ago)
portfolio\weekly_digest.py:271:    config = load_json(CONFIG_FILE, default={})
portfolio\weekly_digest.py:286:    atomic_append_jsonl(log_file, entry)
portfolio\warrant_portfolio.py:31:    state = load_json(WARRANT_STATE_FILE)
portfolio\warrant_portfolio.py:48:    atomic_write_json(WARRANT_STATE_FILE, state)
portfolio\strategies\golddigger_strategy.py:161:        queue = load_json(self._trade_queue_file, default=None)
portfolio\strategies\golddigger_strategy.py:183:        atomic_write_json(self._trade_queue_file, queue)
portfolio\signals\news_event.py:96:        atomic_write_json(_HEADLINES_PATH, payload)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 778ms:
portfolio\accuracy_stats.py:1202:    fh = acquire_lock_file(_DASHBOARD_PREWARM_LOCK_FILE,
portfolio\accuracy_stats.py:1237:            release_lock_file(fh)
portfolio\elongir\config.py:66:    lock_file: str = str(DATA_DIR / "elongir.singleton.lock")
portfolio\elongir\config.py:102:            lock_file=el.get("lock_file", str(DATA_DIR / "elongir.singleton.lock")),
portfolio\elongir\runner.py:83:    fh = acquire_lock_file(
portfolio\elongir\runner.py:98:    release_lock_file(_singleton_lock_fh)
portfolio\fin_snipe_manager.py:52:LOCK_FILE = BASE_DIR / "data" / "fin_snipe_manager.singleton.lock"
portfolio\fin_snipe_manager.py:1687:    lock_handle = acquire_lock_file(
portfolio\fin_snipe_manager.py:1761:        release_lock_file(lock_handle)
scripts\health_check.py:345:    for lock_file in DATA_DIR.glob("*.singleton.lock"):
portfolio\golddigger\runner.py:40:SINGLETON_LOCK_FILE = DATA_DIR / "golddigger.singleton.lock"
portfolio\golddigger\runner.py:104:    fh = acquire_lock_file(
portfolio\golddigger\runner.py:119:    release_lock_file(_singleton_lock_fh)
portfolio\main.py:52:_SINGLETON_LOCK_FILE = str(DATA_DIR / "main_loop.singleton.lock")
portfolio\mstr_loop\config.py:199:SINGLETON_LOCK_FILE = "data/mstr_loop.singleton.lock"
scripts\win\crypto-loop.bat:18:if %EXIT_CODE% EQU 11 (
portfolio\process_lock.py:22:def acquire_lock_file(
portfolio\process_lock.py:47:def release_lock_file(fh: IO[str] | None) -> None:
scripts\win\metals-loop.bat:16:if %EXIT_CODE% EQU 11 (
scripts\win\oil-loop.bat:16:if %EXIT_CODE% EQU 11 (
scripts\win\pf-loop.bat:19:if %EXIT_CODE% EQU 11 (
scripts\win\silver-monitor.bat:12:if %EXIT_CODE% EQU 11 (
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 749ms:
portfolio/process_lock.py:22:def acquire_lock_file(
portfolio\accuracy_stats.py:1202:    fh = acquire_lock_file(_DASHBOARD_PREWARM_LOCK_FILE,
scripts/win\crypto-loop.bat:4:REM instance already holds the singleton lock — we stop instead of
scripts/win\install-crypto-loop-task.ps1:8:#   - Multiple-instance ignored (singleton lock at the Python level)
portfolio\elongir\config.py:66:    lock_file: str = str(DATA_DIR / "elongir.singleton.lock")
portfolio\elongir\config.py:102:            lock_file=el.get("lock_file", str(DATA_DIR / "elongir.singleton.lock")),
portfolio\elongir\runner.py:78:    """Acquire the Elongir singleton lock."""
portfolio\elongir\runner.py:83:    fh = acquire_lock_file(
portfolio\elongir\runner.py:96:    """Release the singleton lock if held."""
portfolio\fin_snipe_manager.py:52:LOCK_FILE = BASE_DIR / "data" / "fin_snipe_manager.singleton.lock"
portfolio\fin_snipe_manager.py:1687:    lock_handle = acquire_lock_file(
scripts/win\oil-loop.bat:4:REM instance already holds the singleton lock — we stop instead of
scripts/win\pf-restart.ps1:8:  keeping the old code in memory and blocking the file-singleton lock
portfolio\golddigger\runner.py:40:SINGLETON_LOCK_FILE = DATA_DIR / "golddigger.singleton.lock"
portfolio\golddigger\runner.py:99:    """Acquire the GoldDigger singleton lock."""
portfolio\golddigger\runner.py:104:    fh = acquire_lock_file(
portfolio\golddigger\runner.py:117:    """Release the GoldDigger singleton lock if held."""
portfolio\main.py:52:_SINGLETON_LOCK_FILE = str(DATA_DIR / "main_loop.singleton.lock")
portfolio\process_lock.py:22:def acquire_lock_file(
portfolio\mstr_loop\config.py:199:SINGLETON_LOCK_FILE = "data/mstr_loop.singleton.lock"
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"send_or_store\\(|log_message\\(|BankID|telegram.*token|dashboard_token|authorization token|auth token|chat_id|otp|TOTP|verification code|session\" portfolio scripts dashboard" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "BankID|bankid|TOTP|otp|session token|dashboard_token|telegram token|chat_id" portfolio/avanza_session.py portfolio/telegram_notifications.py portfolio/message_store.py portfolio/avanza_client.py portfolio/avanza_orders.py portfolio/avanza_control.py' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"message_store|telegram_messages\\.jsonl|send_or_store\\(\" tests portfolio scripts" in Q:\finance-analyzer
 succeeded in 730ms:
portfolio\daily_digest.py:17:from portfolio.message_store import send_or_store
portfolio\daily_digest.py:274:            send_or_store(msg, config, category="daily_digest")
portfolio\digest.py:3:Sends via message_store with category "digest" (always delivered to Telegram).
portfolio\digest.py:16:from portfolio.message_store import send_or_store
portfolio\digest.py:267:        send_or_store(msg, config, category="digest")
portfolio\crypto_scheduler.py:25:from portfolio.message_store import send_or_store
portfolio\crypto_scheduler.py:367:        send_or_store(msg, config, category="crypto_report")
portfolio\autonomous.py:19:from portfolio.message_store import send_or_store
portfolio\autonomous.py:176:            send_or_store(msg, config, category="analysis")
portfolio\agent_invocation.py:17:from portfolio.message_store import send_or_store
portfolio\agent_invocation.py:26:TELEGRAM_FILE = DATA_DIR / "telegram_messages.jsonl"
portfolio\agent_invocation.py:928:            send_or_store(notify_msg, config, category="invocation")
portfolio\agent_invocation.py:1274:            send_or_store(
portfolio\agent_invocation.py:1284:            send_or_store(
portfolio\agent_invocation.py:1309:                send_or_store(
portfolio\bigbet.py:16:from portfolio.message_store import send_or_store
portfolio\bigbet.py:619:    send_or_store(msg, config, category="bigbet")
scripts\health_check.py:34:from portfolio.message_store import send_or_store
scripts\health_check.py:288:    entries = load_jsonl_tail(DATA_DIR / "telegram_messages.jsonl", max_entries=20)
scripts\health_check.py:535:        ok = send_or_store(msg, config, category="health")
portfolio\accuracy_degradation.py:868:        from portfolio.message_store import send_or_store
portfolio\accuracy_degradation.py:869:        # send_or_store(msg, config, category=...) — config is REQUIRED
portfolio\accuracy_degradation.py:874:        send_or_store(body, config or {}, category="daily_digest")
scripts\monitor_silver_exit.py:24:from portfolio.message_store import send_or_store
scripts\monitor_silver_exit.py:72:        send_or_store(msg, _CONFIG, category="analysis")
portfolio\fin_snipe_manager.py:100:        from portfolio.message_store import send_or_store
portfolio\fin_snipe_manager.py:104:            send_or_store(message, config, category="error")
portfolio\focus_analysis.py:3:Provides a one-shot report (console + Telegram via message_store) with:
portfolio\focus_analysis.py:18:from portfolio.message_store import send_or_store
portfolio\focus_analysis.py:235:    send_or_store(msg, config, category="analysis")
portfolio\fx_rates.py:88:        from portfolio.message_store import send_or_store
portfolio\fx_rates.py:89:        send_or_store(msg, config, category="error")
portfolio\elongir\runner.py:70:        from portfolio.message_store import send_or_store
portfolio\elongir\runner.py:71:        return send_or_store(msg, config, category="elongir")
tests\test_accuracy_degradation.py:586:        # Real signature is send_or_store(msg, config, category=...).
tests\test_accuracy_degradation.py:589:            "portfolio.message_store.send_or_store",
tests\test_agent_completion.py:81:    tg_file = tmp_path / "telegram_messages.jsonl"
portfolio\golddigger\runner.py:31:from portfolio.message_store import send_or_store
portfolio\golddigger\runner.py:91:    """Send a Telegram notification via message_store."""
portfolio\golddigger\runner.py:93:        send_or_store(msg, config, category="golddigger")
tests\test_agent_invocation_watchdog.py:142:    telegram = tmp_path / "telegram_messages.jsonl"
tests\test_agent_invocation.py:729:        monkeypatch.setattr(ai, "TELEGRAM_FILE", tmp_data / "telegram_messages.jsonl")
portfolio\iskbets.py:21:from portfolio.message_store import send_or_store
portfolio\iskbets.py:84:    send_or_store(msg, config, category="iskbets")
portfolio\iskbets.py:89:    from portfolio.message_store import log_message
portfolio\log_rotation.py:85:    "telegram_messages.jsonl": {
portfolio\loop_contract.py:91:# change this without updating message_store.SEND_CATEGORIES too.
portfolio\loop_contract.py:2011:    """Mirror message_store.send_or_store's mute gating to predict whether
portfolio\loop_contract.py:2018:    logic in portfolio/message_store.py:170-219 — keep both in sync.
portfolio\loop_contract.py:2021:    which message_store._do_send_telegram uses as a fast-path that
portfolio\loop_contract.py:2231:        from portfolio.message_store import send_or_store
portfolio\loop_contract.py:2238:        sent_ok = send_or_store(msg, config, category=CONTRACT_ALERT_CATEGORY)
tests\test_bug_fixes_session4.py:6:BUG-33: message_store SEND_CATEGORIES includes "invocation" (should be save-only)
tests\test_bug_fixes_session4.py:117:# BUG-33: message_store SEND_CATEGORIES excludes "invocation"
tests\test_bug_fixes_session4.py:124:        from portfolio.message_store import SEND_CATEGORIES
tests\test_bug_fixes_session4.py:128:        from portfolio.message_store import SEND_CATEGORIES
tests\test_bug_fixes_session4.py:132:        from portfolio.message_store import SEND_CATEGORIES
tests\test_batch2_fixes.py:69:    tg_file = tmp_path / "telegram_messages.jsonl"
portfolio\message_throttle.py:102:    from portfolio.message_store import send_or_store
portfolio\message_throttle.py:105:        send_or_store(text, config, category="analysis")
portfolio\message_throttle.py:120:    This is a no-op — trades go through message_store directly.
portfolio\main.py:360:    for name in ("invocations.jsonl", "layer2_journal.jsonl", "telegram_messages.jsonl", "claude_invocations.jsonl"):
portfolio\main.py:771:            from portfolio.message_store import send_or_store
portfolio\main.py:772:            send_or_store(_fail_msg, config, category="error")
portfolio\main.py:935:                    from portfolio.message_store import send_or_store
portfolio\main.py:936:                    send_or_store(msg, config, category="error")
portfolio\main.py:947:                    from portfolio.message_store import send_or_store
portfolio\main.py:948:                    send_or_store(msg, config, category="error")
portfolio\main.py:1020:                from portfolio.message_store import send_or_store
portfolio\main.py:1021:                send_or_store(text, config, category="error")
portfolio\main.py:1032:        from portfolio.message_store import send_or_store
portfolio\main.py:1033:        send_or_store(text, config, category="error")
portfolio\main.py:1155:                    from portfolio.message_store import send_or_store
portfolio\main.py:1156:                    send_or_store(msg, config, category="error")
portfolio\message_store.py:30:logger = logging.getLogger("portfolio.message_store")
portfolio\message_store.py:33:MESSAGES_FILE = BASE_DIR / "data" / "telegram_messages.jsonl"
portfolio\message_store.py:170:def send_or_store(msg, config, category="analysis"):
tests\test_bug_fixes_session_20260321.py:321:            with patch("portfolio.message_store.send_or_store") as mock_send:
tests\test_bug_fixes_session_20260321.py:336:            with patch("portfolio.message_store.send_or_store") as mock_send:
portfolio\mstr_loop\telegram_report.py:3:Thin wrapper around portfolio.message_store.send_or_store. Throttling via
portfolio\mstr_loop\telegram_report.py:139:    """Backend send — routes through portfolio.message_store.send_or_store."""
portfolio\mstr_loop\telegram_report.py:142:        from portfolio.message_store import send_or_store
portfolio\mstr_loop\telegram_report.py:144:        send_or_store(text, cfg, category=category)
tests\test_dashboard.py:265:        (tmp_data / "telegram_messages.jsonl").write_text('{"ts":"t","text":"hi"}\n', encoding="utf-8")
tests\test_dashboard.py:287:        (tmp_data / "telegram_messages.jsonl").write_text(
tests\test_dashboard.py:378:        (tmp_data / "telegram_messages.jsonl").write_text(
tests\test_dashboard.py:388:        (tmp_data / "telegram_messages.jsonl").write_text(
tests\test_dashboard.py:400:        (tmp_data / "telegram_messages.jsonl").write_text(
portfolio\regime_alerts.py:152:    Also logs the message to telegram_messages.jsonl.
portfolio\regime_alerts.py:180:        from portfolio.message_store import send_or_store
portfolio\regime_alerts.py:181:        send_or_store(msg, config, category="regime")
portfolio\stats.py:9:TELEGRAMS_FILE = DATA_DIR / "telegram_messages.jsonl"
portfolio\telegram_notifications.py:9:from portfolio.message_store import send_or_store
portfolio\telegram_notifications.py:139:        send_or_store(msg, config, category="analysis")
portfolio\weekly_digest.py:280:    log_file = DATA_DIR / "telegram_messages.jsonl"
tests\test_digest.py:229:        telegram_log = data_dir / "telegram_messages.jsonl"
tests\test_fin_snipe_manager.py:971:        "portfolio.message_store.send_or_store",
tests\test_headless_env_var.py:41:        import portfolio.message_store as ms
tests\test_loop_contract_alert_cooldown.py:51:        with patch("portfolio.message_store.send_or_store") as mock_send:
tests\test_loop_contract_alert_cooldown.py:64:        with patch("portfolio.message_store.send_or_store") as mock_send:
tests\test_loop_contract_alert_cooldown.py:85:        with patch("portfolio.message_store.send_or_store") as mock_send:
tests\test_loop_contract_alert_cooldown.py:99:        with patch("portfolio.message_store.send_or_store") as mock_send:
tests\test_loop_contract_alert_cooldown.py:130:        with patch("portfolio.message_store.send_or_store") as mock_send:
tests\test_loop_contract_alert_cooldown.py:150:        with patch("portfolio.message_store.send_or_store") as mock_send:
tests\test_loop_contract_alert_cooldown.py:171:        with patch("portfolio.message_store.send_or_store"):
tests\test_loop_contract_alert_cooldown.py:191:        with patch("portfolio.message_store.send_or_store") as mock_send:
tests\test_loop_contract_alert_cooldown.py:203:        with patch("portfolio.message_store.send_or_store") as mock_send:
tests\test_loop_contract_alert_cooldown.py:215:        message_store._do_send_telegram to return True without delivering.
tests\test_loop_contract_alert_cooldown.py:229:        with patch("portfolio.message_store.send_or_store", return_value=True):
tests\test_loop_contract_alert_cooldown.py:253:        with patch("portfolio.message_store.send_or_store") as mock_send:
tests\test_loop_contract_alert_cooldown.py:281:        with patch("portfolio.message_store.send_or_store"):
tests\test_loop_contract_alert_cooldown.py:311:            "portfolio.message_store.send_or_store",
tests\test_loop_contract_alert_cooldown.py:340:            "portfolio.message_store.send_or_store",
tests\test_loop_contract_alert_cooldown.py:373:            "portfolio.message_store.send_or_store",
tests\test_loop_contract_alert_cooldown.py:403:            "portfolio.message_store.send_or_store",
tests\test_loop_contract_alert_cooldown.py:445:        with patch("portfolio.message_store.send_or_store") as mock_send:
tests\test_loop_contract_alert_cooldown.py:480:        with patch("portfolio.message_store.send_or_store") as mock_send:
tests\test_loop_contract_alert_cooldown.py:534:            "portfolio.message_store.send_or_store",
tests\test_loop_contract_alert_cooldown.py:584:        with patch("portfolio.message_store.send_or_store") as mock_send:
tests\test_loop_contract_alert_cooldown.py:610:            "portfolio.message_store.send_or_store",
tests\test_message_store.py:1:"""Tests for portfolio.message_store readability hardening + truncation safety."""
tests\test_message_store.py:8:    from portfolio.message_store import sanitize_message_text
tests\test_message_store.py:27:    from portfolio.message_store import sanitize_message_text
tests\test_message_store.py:37:    import portfolio.message_store as message_store
tests\test_message_store.py:40:    log_path = tmp_path / "telegram_messages.jsonl"
tests\test_message_store.py:42:    monkeypatch.setattr(message_store, "MESSAGES_FILE", log_path)
tests\test_message_store.py:44:        message_store,
tests\test_message_store.py:50:    result = message_store.send_or_store(
tests\test_message_store.py:80:        import portfolio.message_store as ms
tests\test_message_store.py:117:        import portfolio.message_store as ms
tests\test_message_store.py:142:        import portfolio.message_store as ms
tests\test_message_throttle.py:116:    @patch("portfolio.message_store.send_or_store")
tests\test_resilience_improvements.py:118:    @mock.patch("portfolio.message_store.send_or_store")
tests\test_resilience_improvements.py:140:            with mock.patch("portfolio.message_store.send_or_store") as mock_send:
tests\test_resilience_improvements.py:144:            pass  # message_store may fail to import in test env
tests\test_resilience_improvements.py:168:        for name in ("invocations.jsonl", "layer2_journal.jsonl", "telegram_messages.jsonl"):
tests\test_resilience_improvements.py:178:        assert "telegram_messages.jsonl" not in _prune_failures
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 740ms:
portfolio/avanza_control.py:260:# --- Page-free API (uses BankID session, no Playwright page needed) ---
portfolio/avanza_control.py:289:    """Fetch a quote without a Playwright page — uses BankID session API."""
portfolio/avanza_control.py:335:    """Place BUY/SELL via BankID session — no Playwright page needed.
portfolio/avanza_control.py:357:    """Hardware stop-loss via BankID session — no Playwright page needed.
portfolio/avanza_control.py:368:    """Hardware trailing stop via BankID session — no Playwright page needed.
portfolio/avanza_control.py:379:    """Cancel order via BankID session — no Playwright page needed.
portfolio/avanza_control.py:390:    """Delete stop-loss via BankID session — no Playwright page needed.
portfolio/avanza_orders.py:165:        config: App config dict (with telegram.token, telegram.chat_id,
portfolio/avanza_orders.py:236:        deliver fake updates with the right ``chat_id`` and execute orders.
portfolio/avanza_orders.py:246:    chat_id = str(config.get("telegram", {}).get("chat_id", ""))
portfolio/avanza_orders.py:247:    if not token or not chat_id:
portfolio/avanza_orders.py:295:        if str(msg.get("chat", {}).get("id")) != chat_id:
portfolio/avanza_orders.py:307:                    sender_id, allowed_user, chat_id,
portfolio/avanza_client.py:4:1. BankID session (preferred) — captured by scripts/avanza_login.py, stored in
portfolio/avanza_client.py:6:2. TOTP credentials (fallback) — uses avanza-api library with username/password/TOTP
portfolio/avanza_client.py:9:The client transparently tries BankID session first, then falls back to TOTP.
portfolio/avanza_client.py:25:# A-AV-2 (2026-04-11): Hardcoded account whitelist. The TOTP path scans for
portfolio/avanza_client.py:36:# Cached signal that a BankID Playwright session has already been verified.
portfolio/avanza_client.py:44:        dict with keys: username, password, totp_secret
portfolio/avanza_client.py:56:            "Add: {\"avanza\": {\"username\": \"...\", \"password\": \"...\", \"totp_secret\": \"...\"}}"
portfolio/avanza_client.py:59:    for key in ("username", "password", "totp_secret"):
portfolio/avanza_client.py:66:    """Return True when a BankID-backed Playwright session is available."""
portfolio/avanza_client.py:74:            logger.info("Using BankID session for Avanza API")
portfolio/avanza_client.py:76:        logger.info("BankID session exists but verification failed")
portfolio/avanza_client.py:78:        logger.debug("BankID session not available: %s", e)
portfolio/avanza_client.py:85:    Tries BankID session first, then falls back to TOTP credentials.
portfolio/avanza_client.py:106:        "totpSecret": creds["totp_secret"],
portfolio/avanza_client.py:112:    """Reset the singleton TOTP client (useful for re-authentication)."""
portfolio/avanza_client.py:118:    """Reset the cached BankID session verification flag."""
portfolio/avanza_client.py:140:    Tries BankID session first, then falls back to TOTP client.
portfolio/avanza_client.py:154:            logger.warning("Session-based price fetch failed, trying TOTP: %s", e)
portfolio/avanza_client.py:165:    Tries BankID session first, then falls back to TOTP client.
portfolio/avanza_client.py:177:            logger.warning("Session-based positions fetch failed, trying TOTP: %s", e)
portfolio/avanza_client.py:330:    P0-4 (2026-05-02): Wrapped in ``avanza_order_lock`` so the TOTP path
portfolio/avanza_client.py:339:    (``place_order_totp/...``) so the rate-limit diagnostic ("which loop
portfolio/avanza_client.py:358:    with avanza_order_lock(op=f"place_order_totp/{order_type.value}/{orderbook_id}"):
portfolio/avanza_client.py:396:    with avanza_order_lock(op=f"delete_order_totp/{order_id}"):
portfolio/message_store.py:118:    chat_id = config.get("telegram", {}).get("chat_id")
portfolio/message_store.py:119:    if not token or not chat_id:
portfolio/message_store.py:120:        logger.warning("Telegram token/chat_id not configured")
portfolio/message_store.py:140:        json_body={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
portfolio/message_store.py:163:                json_body={"chat_id": chat_id, "text": msg},
portfolio/message_store.py:181:        config: Full config dict (needs ``telegram.token`` and ``telegram.chat_id``).
portfolio/telegram_notifications.py:53:    chat_id = config["telegram"]["chat_id"]
portfolio/telegram_notifications.py:57:        json_body={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
portfolio/telegram_notifications.py:77:                json_body={"chat_id": chat_id, "text": msg},
portfolio/avanza_session.py:1:"""Avanza session management — load, validate, and use BankID-captured sessions.
portfolio/avanza_session.py:7:This is the preferred auth method until TOTP credentials are configured.
portfolio/avanza_session.py:65:    """Load saved BankID session metadata from disk.
portfolio/avanza_session.py:201:# mid-flight (OS sleep, memory pressure, external BankID re-auth by the
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 759ms:
dashboard\export_static.py:61:def _get_dashboard_token():
dashboard\export_static.py:67:    return cfg.get("dashboard_token") or None
dashboard\export_static.py:82:    token = _get_dashboard_token()
dashboard\auth.py:18:Tests that patched `dashboard.app._get_dashboard_token` or
dashboard\auth.py:45:COOKIE_NAME = "pf_dashboard_token"
dashboard\auth.py:81:def _get_dashboard_token() -> str | None:
dashboard\auth.py:82:    """Return the configured dashboard_token, or None if not set."""
dashboard\auth.py:83:    return _get_config().get("dashboard_token") or None
dashboard\auth.py:109:      1. Cookie (`pf_dashboard_token`) — for repeat visits.
dashboard\auth.py:116:    If no dashboard_token is configured, access is allowed (backwards
dashboard\auth.py:121:        expected = _get_dashboard_token()
dashboard\auth.py:159:                "cookie. Replace YOUR_TOKEN with the dashboard_token from "
dashboard\trading_status.py:45:# Avanza trading session in Europe/Stockholm — DST handled by zoneinfo.
dashboard\trading_status.py:48:# even though Elongir's actual config session is 08:30–21:30. The old
dashboard\trading_status.py:70:        "session_open": _in_session(now),
dashboard\trading_status.py:104:    if not _in_session(now):
dashboard\trading_status.py:112:        "in session, no entry signal yet",
dashboard\trading_status.py:136:    if not _in_session(now):
dashboard\trading_status.py:144:        "in session, no dip detected",
dashboard\trading_status.py:162:    if not _in_session(now):
dashboard\trading_status.py:170:    reason = "in session, no signal"
dashboard\trading_status.py:172:        reason = f"in session, {consecutive_losses} consecutive losses (caution)"
dashboard\trading_status.py:174:        reason = "in session, between trades"
dashboard\trading_status.py:212:    if not _in_session(now):
dashboard\trading_status.py:277:def _in_session(now_utc: datetime) -> bool:
dashboard\trading_status.py:279:    session (Mon–Fri, 15:30–21:55 inclusive of open, exclusive of close).
dashboard\trading_status.py:282:    Sunday at 16:00 local time would otherwise read as session_open and
dashboard\trading_status.py:299:    which lied to users once the session moved to 08:30.
dashboard\app.py:333:    state["session_active"] = (
dashboard\app.py:745:    _get_dashboard_token,
dashboard\app.py:779:    """Clear the pf_dashboard_token cookie and redirect to /.
dashboard\app.py:794:        "pf_dashboard_token",
dashboard\app.py:1704:# Solution: a single dedicated worker thread owns the Playwright session
dashboard\app.py:1784:    Uses `portfolio.avanza_session` (Playwright BankID auth at
dashboard\app.py:1785:    `data/avanza_session.json`) — the same path the live metals_loop and
dashboard\app.py:1786:    golddigger use. The newer `portfolio.avanza` TOTP package is *not*
dashboard\app.py:1787:    used here because TOTP credentials aren't populated in the live
dashboard\app.py:1789:    2026-05-04 originally seeded the TOTP singleton, but the empty
dashboard\app.py:1807:        from portfolio.avanza_session import DEFAULT_ACCOUNT_ID
dashboard\app.py:1814:        from portfolio.avanza_session import get_buying_power
dashboard\app.py:1819:                "(Avanza session likely expired — re-auth via BankID)"
dashboard\app.py:1827:        from portfolio.avanza_session import get_positions
dashboard\app.py:1837:        from portfolio.avanza_session import get_open_orders
dashboard\app.py:1843:        from portfolio.avanza_session import get_stop_losses
dashboard\app.py:2044:# (data/findapartments runs + innerstad heatmap). Reuses pf_dashboard_token
dashboard\house_blueprint.py:6:protected by the same `pf_dashboard_token` that gates the finance dashboard.
dashboard\system_status.py:206:    regression already disposition'd by the 2026-05-03 research session).
scripts\auto-improve.bat:3:REM  PF-AutoImprove — Daily autonomous improvement session
scripts\auto-improve.bat:6:REM  Progress: data\auto-improve-progress.json (written by Claude during session)
scripts\auto-improve.bat:16:REM --- Reset progress file for new session ---
scripts\auto-improve.bat:18:  "Set-Content -Path 'Q:\finance-analyzer\data\auto-improve-progress.json' -Value ('{\"session_started\":\"%TS_START%\",\"current_phase\":\"starting\",\"status\":\"starting\",\"phases_completed\":[],\"notes\":\"Session launched, waiting for Claude to begin\"}')"
scripts\auto-improve.bat:20:REM --- Log: session starting ---
scripts\auto-improve.bat:24:echo [%TS_START%] AutoImprove session starting...
scripts\auto-improve-codex.bat:21:powershell -NoProfile -Command "Set-Content -Path '%PROGRESS_FILE%' -Value ('{\"session_started\":\"%TS_START%\",\"current_phase\":\"startup\",\"status\":\"failed\",\"phases_completed\":[],\"notes\":\"No usable Codex model found.\"}')"
scripts\auto-improve-codex.bat:27:powershell -NoProfile -Command "Set-Content -Path '%PROGRESS_FILE%' -Value ('{\"session_started\":\"%TS_START%\",\"current_phase\":\"starting\",\"status\":\"starting\",\"phases_completed\":[],\"notes\":\"Session launched with %CODEX_MODEL%, waiting for Codex to begin\",\"model\":\"%CODEX_MODEL%\"}')"
scripts\after-hours-research.bat:27:  "Set-Content -Path 'Q:\finance-analyzer\data\after-hours-research-progress.json' -Value ('{\"session_started\":\"%TS_START%\",\"current_phase\":\"starting\",\"status\":\"starting\",\"phases_completed\":[],\"notes\":\"Session launched, waiting for Claude to begin\"}')"
scripts\after-hours-research.bat:29:REM --- Log: session starting ---
scripts\after-hours-research.bat:33:echo [%TS_START%] After-Hours Research session starting...
scripts\avanza_metals_check.py:67:        from portfolio.avanza_session import get_positions, verify_session
scripts\avanza_metals_check.py:69:        if not verify_session():
scripts\avanza_metals_check.py:70:            result["error"] = "Avanza session expired or invalid. Run: python scripts/avanza_login.py"
scripts\avanza_smoke_test.py:4:Requires valid TOTP credentials in config.json.
scripts\avanza_smoke_test.py:18:print("1. Authenticating with TOTP...")
portfolio\avanza_orders.py:165:        config: App config dict (with telegram.token, telegram.chat_id,
portfolio\avanza_orders.py:236:        deliver fake updates with the right ``chat_id`` and execute orders.
portfolio\avanza_orders.py:245:    token = config.get("telegram", {}).get("token", "")
portfolio\avanza_orders.py:246:    chat_id = str(config.get("telegram", {}).get("chat_id", ""))
portfolio\avanza_orders.py:247:    if not token or not chat_id:
portfolio\avanza_orders.py:273:            f"https://api.telegram.org/bot{token}/getUpdates",
portfolio\avanza_orders.py:295:        if str(msg.get("chat", {}).get("id")) != chat_id:
portfolio\avanza_orders.py:307:                    sender_id, allowed_user, chat_id,
scripts\avanza_orders.py:7:Requires Avanza credentials or BankID session (see scripts/avanza_login.py).
scripts\avanza_login.py:1:"""Avanza BankID login via Playwright browser automation.
scripts\avanza_login.py:4:to authenticate via BankID on their phone, then captures the session cookies
scripts\avanza_login.py:10:The session is saved to data/avanza_session.json and is valid for ~24 hours.
scripts\avanza_login.py:21:SESSION_FILE = DATA_DIR / "avanza_session.json"
scripts\avanza_login.py:31:    """Check if user has authenticated via BankID.
scripts\avanza_login.py:34:    csid+cstoken cookies (only set after BankID authentication).
scripts\avanza_login.py:41:    # Strong signal: csid + cstoken cookies (only set after BankID)
scripts\avanza_login.py:50:    """Launch browser, wait for BankID auth, capture session."""
scripts\avanza_login.py:62:    print("2. Authenticate with BankID on your phone")
scripts\avanza_login.py:63:    print("3. The script will detect login automatically and save your session")
scripts\avanza_login.py:82:                    if any(t in k.lower() for t in ("securi", "auth", "token", "session", "csrf", "cookie"))
scripts\avanza_login.py:89:                        # Capture request headers as the "real" auth tokens
scripts\avanza_login.py:93:                        elif kl in ("x-authenticationsession", "x-authenticationssession"):
scripts\avanza_login.py:94:                            captured_tokens["authentication_session"] = v
scripts\avanza_login.py:109:            if "x-authenticationsession" in lower_headers and "authentication_session" not in captured_tokens:
scripts\avanza_login.py:110:                captured_tokens["authentication_session"] = lower_headers[
scripts\avanza_login.py:111:                    "x-authenticationsession"
scripts\avanza_login.py:114:            # Check response body for auth session info
scripts\avanza_login.py:115:            if "_api" in url and ("auth" in url.lower() or "session" in url.lower()):
scripts\avanza_login.py:119:                        for key in ("authenticationSession", "authentication_session"):
scripts\avanza_login.py:121:                                captured_tokens["authentication_session"] = body[key]
scripts\avanza_login.py:158:        print("Browser opened. Click 'Logga in' and authenticate with BankID.")
scripts\avanza_login.py:245:    session_data = {
scripts\avanza_login.py:248:        "authentication_session": captured_tokens.get("authentication_session"),
scripts\avanza_login.py:255:    # Save session
scripts\avanza_login.py:258:        json.dumps(session_data, indent=2, ensure_ascii=False), encoding="utf-8"
scripts\avanza_login.py:263:    print(f"  Security token: {'YES' if session_data['security_token'] else 'NO (will try cookie-based auth)'}")
scripts\avanza_login.py:264:    print(f"  Authentication session: {'YES' if session_data['authentication_session'] else 'NO'}")
scripts\avanza_login.py:265:    print(f"  Customer ID: {session_data['customer_id'] or 'unknown'}")
scripts\avanza_login.py:271:    return session_data
portfolio\agent_invocation.py:841:        # Strip Claude Code session markers to avoid "nested session" error
portfolio\agent_invocation.py:928:            send_or_store(notify_msg, config, category="invocation")
portfolio\agent_invocation.py:1274:            send_or_store(
portfolio\agent_invocation.py:1284:            send_or_store(
portfolio\agent_invocation.py:1309:                send_or_store(
portfolio\analyze.py:24:    """Return env dict without CLAUDECODE to avoid nested-session errors.
portfolio\analyze.py:295:        # helps future Claude sessions notice when auth expires.
portfolio\avanza_control.py:1:"""Canonical Avanza control facade for reads, quotes, and browser-session trades.
portfolio\avanza_control.py:5:while exposing the broader account/session helpers from ``portfolio.avanza_*``.
portfolio\avanza_control.py:18:    check_session_alive,
portfolio\avanza_control.py:113:    """Fetch buying power for an account via the authenticated browser session."""
portfolio\avanza_control.py:119:    """Fetch current positions keyed by orderbook id via the page session.
portfolio\avanza_control.py:131:    """Place a BUY/SELL order via the authenticated browser session."""
portfolio\avanza_control.py:146:    """Place a hardware stop-loss order via the authenticated browser session."""
portfolio\avanza_control.py:160:    """Cancel an open order via the authenticated page session.
portfolio\avanza_control.py:260:# --- Page-free API (uses BankID session, no Playwright page needed) ---
portfolio\avanza_control.py:262:from portfolio.avanza_session import (
portfolio\avanza_control.py:265:from portfolio.avanza_session import (
portfolio\avanza_control.py:268:from portfolio.avanza_session import (
portfolio\avanza_control.py:271:from portfolio.avanza_session import (
portfolio\avanza_control.py:274:from portfolio.avanza_session import (
portfolio\avanza_control.py:277:from portfolio.avanza_session import (
portfolio\avanza_control.py:278:    place_stop_loss as _place_stop_loss_session,
portfolio\avanza_control.py:280:from portfolio.avanza_session import (
portfolio\avanza_control.py:281:    place_trailing_stop as _place_trailing_stop_session,
portfolio\avanza_control.py:283:from portfolio.avanza_session import (
portfolio\avanza_control.py:284:    verify_session,
portfolio\avanza_control.py:289:    """Fetch a quote without a Playwright page — uses BankID session API."""
portfolio\avanza_control.py:335:    """Place BUY/SELL via BankID session — no Playwright page needed.
portfolio\avanza_control.py:357:    """Hardware stop-loss via BankID session — no Playwright page needed.
portfolio\avanza_control.py:362:    result = _place_stop_loss_session(ob_id, trigger_price, sell_price, volume, account_id, valid_days)
portfolio\avanza_control.py:368:    """Hardware trailing stop via BankID session — no Playwright page needed.
portfolio\avanza_control.py:373:    result = _place_trailing_stop_session(ob_id, trail_percent, volume, account_id, valid_days)
portfolio\avanza_control.py:379:    """Cancel order via BankID session — no Playwright page needed.
portfolio\avanza_control.py:390:    """Delete stop-loss via BankID session — no Playwright page needed.
portfolio\avanza_control.py:411:    "check_session_alive",
portfolio\avanza_control.py:438:    "verify_session",
portfolio\api_utils.py:53:    return tg.get("token", ""), tg.get("chat_id", "")
portfolio\avanza_account_check.py:5:``portfolio/avanza_session.py``) showed Beammwave / NextEra / Vertiv
portfolio\avanza_account_check.py:74:    ``avanza_session.get_buying_power`` uses so we cover shape drift.
portfolio\avanza_account_check.py:99:    order as ``avanza_session.get_buying_power``."""
portfolio\avanza_account_check.py:148:        if cfg.get("telegram", {}).get("token"):
portfolio\avanza_account_check.py:155:    """Pulled out so tests can mock without touching avanza_session
portfolio\avanza_account_check.py:165:    ``GridFisher._safe_session_call`` uses.
portfolio\avanza_account_check.py:169:    from portfolio.avanza_session import api_get  # noqa: PLC0415
portfolio\avanza_account_check.py:198:            ``avanza_session.DEFAULT_ACCOUNT_ID`` so most callers can
portfolio\avanza_account_check.py:211:        from portfolio.avanza_session import DEFAULT_ACCOUNT_ID  # noqa: PLC0415
scripts\check_critical_errors.py:3:Invoked at the start of every Claude Code session in this project (via
scripts\check_critical_errors.py:15:  is healthy; non-zero output is compact enough to fit in a session's
scripts\check_critical_errors.py:74:        # rolling noise on every Claude session start.
portfolio\accuracy_stats.py:1580:    malformed lines at debug level, so torn writes leave a footprint.
portfolio\avanza_client.py:4:1. BankID session (preferred) — captured by scripts/avanza_login.py, stored in
portfolio\avanza_client.py:5:   data/avanza_session.json. No credentials needed, valid ~24h.
portfolio\avanza_client.py:6:2. TOTP credentials (fallback) — uses avanza-api library with username/password/TOTP
portfolio\avanza_client.py:9:The client transparently tries BankID session first, then falls back to TOTP.
portfolio\avanza_client.py:25:# A-AV-2 (2026-04-11): Hardcoded account whitelist. The TOTP path scans for
portfolio\avanza_client.py:31:# avanza_session.py: anything not in this set is rejected, period.
portfolio\avanza_client.py:36:# Cached signal that a BankID Playwright session has already been verified.
portfolio\avanza_client.py:37:_session_client = None
portfolio\avanza_client.py:44:        dict with keys: username, password, totp_secret
portfolio\avanza_client.py:56:            "Add: {\"avanza\": {\"username\": \"...\", \"password\": \"...\", \"totp_secret\": \"...\"}}"
portfolio\avanza_client.py:59:    for key in ("username", "password", "totp_secret"):
portfolio\avanza_client.py:65:def _try_session_auth() -> bool:
portfolio\avanza_client.py:66:    """Return True when a BankID-backed Playwright session is available."""
portfolio\avanza_client.py:67:    global _session_client
portfolio\avanza_client.py:68:    if _session_client is True:
portfolio\avanza_client.py:71:        from portfolio.avanza_session import verify_session
portfolio\avanza_client.py:72:        if verify_session():
portfolio\avanza_client.py:73:            _session_client = True
portfolio\avanza_client.py:74:            logger.info("Using BankID session for Avanza API")
portfolio\avanza_client.py:76:        logger.info("BankID session exists but verification failed")
portfolio\avanza_client.py:78:        logger.debug("BankID session not available: %s", e)
portfolio\avanza_client.py:85:    Tries BankID session first, then falls back to TOTP credentials.
portfolio\avanza_client.py:106:        "totpSecret": creds["totp_secret"],
portfolio\avanza_client.py:112:    """Reset the singleton TOTP client (useful for re-authentication)."""
portfolio\avanza_client.py:117:def reset_session() -> None:
portfolio\avanza_client.py:118:    """Reset the cached BankID session verification flag."""
portfolio\avanza_client.py:119:    global _session_client
portfolio\avanza_client.py:120:    _session_client = None
portfolio\avanza_client.py:140:    Tries BankID session first, then falls back to TOTP client.
portfolio\avanza_client.py:148:    # Try session-based auth first
portfolio\avanza_client.py:149:    if _try_session_auth():
portfolio\avanza_client.py:151:            from portfolio.avanza_session import get_instrument_price
portfolio\avanza_client.py:154:            logger.warning("Session-based price fetch failed, trying TOTP: %s", e)
portfolio\avanza_client.py:155:            reset_session()
portfolio\avanza_client.py:165:    Tries BankID session first, then falls back to TOTP client.
portfolio\avanza_client.py:171:    # Try session-based auth first
portfolio\avanza_client.py:172:    if _try_session_auth():
portfolio\avanza_client.py:174:            from portfolio.avanza_session import get_positions as session_get_positions
portfolio\avanza_client.py:175:            return session_get_positions()
portfolio\avanza_client.py:177:            logger.warning("Session-based positions fetch failed, trying TOTP: %s", e)
portfolio\avanza_client.py:178:            reset_session()
portfolio\avanza_client.py:330:    P0-4 (2026-05-02): Wrapped in ``avanza_order_lock`` so the TOTP path
portfolio\avanza_client.py:331:    cannot race against the page-session paths in
portfolio\avanza_client.py:333:    ``portfolio/avanza_session.place_order``, or
portfolio\avanza_client.py:338:    The op label is distinct from page-session labels
portfolio\avanza_client.py:339:    (``place_order_totp/...``) so the rate-limit diagnostic ("which loop
portfolio\avanza_client.py:358:    with avanza_order_lock(op=f"place_order_totp/{order_type.value}/{orderbook_id}"):
portfolio\avanza_client.py:396:    with avanza_order_lock(op=f"delete_order_totp/{order_id}"):
scripts\fin_fish_monitor.py:28:from portfolio.avanza_session import (
scripts\fin_fish_monitor.py:33:    verify_session,
scripts\fin_fish_monitor.py:89:# Avanza helpers (using avanza_session API functions)
scripts\fin_fish_monitor.py:147:      - portfolio/avanza_session.cancel_stop_loss (line 911)
scripts\fin_fish_monitor.py:152:        from portfolio.avanza_session import api_delete
scripts\fin_fish_monitor.py:232:        from portfolio.avanza_session import place_sell_order
portfolio\accuracy_degradation.py:869:        # send_or_store(msg, config, category=...) — config is REQUIRED
portfolio\accuracy_degradation.py:870:        # (needs telegram.token + telegram.chat_id). Production bug found
portfolio\accuracy_degradation.py:874:        send_or_store(body, config or {}, category="daily_digest")
portfolio\bigbet.py:619:    send_or_store(msg, config, category="bigbet")
portfolio\claude_gate.py:53:# session must see. Intentionally separate from claude_invocations.jsonl so
portfolio\claude_gate.py:56:# referenced from CLAUDE.md to guarantee surfacing at session start.
portfolio\claude_gate.py:97:    """Return a copy of ``os.environ`` with Claude session markers removed.
portfolio\claude_gate.py:99:    Prevents the "nested session" error when invoking ``claude -p`` from a
portfolio\claude_gate.py:100:    process tree that already has a Claude Code session active.
portfolio\claude_gate.py:121:# unresolved critical_errors.jsonl entries verbatim at session start. Those
portfolio\claude_gate.py:171:    ``scripts/check_critical_errors.py`` at Claude session start (via
portfolio\claude_gate.py:173:    future Claude session until it's resolved with a follow-up entry.
portfolio\claude_gate.py:207:    future Claude sessions see it via the CLAUDE.md startup check. Callers
portfolio\claude_gate.py:238:                "OAuth session not being read. Likely causes: "
portfolio\claude_gate.py:248:                    f"claude CLI subprocess printed {marker!r} — OAuth session "
portfolio\claude_gate.py:287:# session this leaks file handles, sockets, and (worst) GPU VRAM held by
portfolio\claude_gate.py:290:# Fix: explicitly Popen with a new process group/session so we can kill the
portfolio\claude_gate.py:293:# process group started via start_new_session=True.
portfolio\claude_gate.py:298:    return {"start_new_session": True}
scripts\fish_monitor_live.py:344:        from portfolio.avanza_session import get_positions
scripts\fish_monitor_live.py:371:        from portfolio.avanza_session import get_quote, place_sell_order
scripts\fish_monitor_live.py:423:        from portfolio.avanza_session import get_buying_power, get_quote, place_buy_order
scripts\fish_monitor_live.py:441:        # None here means Avanza API shape drift or session dead — fail loud
scripts\fish_monitor_live.py:445:            log_msg('SKIP: get_buying_power() returned None — Avanza session may need refresh')
scripts\fish_monitor_live.py:487:    Works at any time of day — looks at recent data, not session start.
scripts\fish_monitor_live.py:561:    session_pnl = 0
scripts\fish_monitor_live.py:608:    log_msg(f'=== FISH MONITOR LIVE | {session_pnl:+.0f} SEK | {mode.upper()} mode | until {SESSION_END_H}:{SESSION_END_M:02d} ===')
scripts\fish_monitor_live.py:632:                        session_pnl += pnl
scripts\fish_monitor_live.py:634:                log_msg(f'Session end: {session_pnl:+.0f} SEK')
scripts\fish_monitor_live.py:648:            is_us_session = 14 <= h < 17  # 14:00-17:00 CET = best hours
scripts\fish_monitor_live.py:744:                        session_pnl += pnl
scripts\fish_monitor_live.py:745:                        log_msg(f'Session: {session_pnl:+.0f} SEK')
scripts\fish_monitor_live.py:912:                    tz = ' [DEAD-ZONE]' if is_dead_zone else (' [US-SESSION]' if is_us_session else '')
portfolio\avanza_session.py:1:"""Avanza session management — load, validate, and use BankID-captured sessions.
portfolio\avanza_session.py:4:headless browser context. This ensures cookies and TLS session match what
portfolio\avanza_session.py:7:This is the preferred auth method until TOTP credentials are configured.
portfolio\avanza_session.py:23:logger = logging.getLogger("portfolio.avanza_session")
portfolio\avanza_session.py:27:SESSION_FILE = DATA_DIR / "avanza_session.json"
portfolio\avanza_session.py:31:# Minimum remaining session life before we consider it expired (minutes)
portfolio\avanza_session.py:61:    """Raised when session is missing, expired, or invalid."""
portfolio\avanza_session.py:64:def load_session() -> dict:
portfolio\avanza_session.py:65:    """Load saved BankID session metadata from disk.
portfolio\avanza_session.py:75:            f"No session file found at {SESSION_FILE}. "
portfolio\avanza_session.py:81:        raise AvanzaSessionError(f"Failed to read session file: {SESSION_FILE}")
portfolio\avanza_session.py:106:def session_remaining_minutes() -> float | None:
portfolio\avanza_session.py:107:    """Get minutes remaining on the current session, or None if no session."""
portfolio\avanza_session.py:119:        logger.warning("Failed to compute session minutes remaining: %s", e)
portfolio\avanza_session.py:123:def is_session_expiring_soon(threshold_minutes: float = 60.0) -> bool:
portfolio\avanza_session.py:124:    """Check if session will expire within the given threshold.
portfolio\avanza_session.py:126:    Returns True if session is expired, expiring soon, or doesn't exist.
portfolio\avanza_session.py:128:    remaining = session_remaining_minutes()
portfolio\avanza_session.py:142:        # Validate session first
portfolio\avanza_session.py:143:        load_session()
portfolio\avanza_session.py:180:def verify_session() -> bool:
portfolio\avanza_session.py:181:    """Verify that the session is valid by making a lightweight API call.
portfolio\avanza_session.py:184:        True if session is valid, False otherwise.
portfolio\avanza_session.py:201:# mid-flight (OS sleep, memory pressure, external BankID re-auth by the
portfolio\avanza_session.py:227:                "avanza_session: browser dead on %s (%r) — teardown + relaunch + retry",
portfolio\avanza_session.py:248:        AvanzaSessionError: if session is invalid.
portfolio\avanza_session.py:283:        raise AvanzaSessionError("No AZACSRF cookie found — session may be invalid")
portfolio\avanza_session.py:291:        raise AvanzaSessionError("No AZACSRF cookie found — session may be invalid")
portfolio\avanza_session.py:677:    """Get all positions via session-based auth.
scripts\fish_preflight.py:17:Designed to run BEFORE every /fin-fish session as a gate.
portfolio\autonomous.py:176:            send_or_store(msg, config, category="analysis")
portfolio\config_validator.py:24:    ("telegram", "token"),
portfolio\config_validator.py:25:    ("telegram", "chat_id"),
portfolio\avanza_tracker.py:107:def check_session_expiry() -> str | None:
portfolio\avanza_tracker.py:108:    """Check if Avanza BankID session is expired or expiring soon.
portfolio\avanza_tracker.py:111:        Warning message string if session needs refresh, None if OK.
portfolio\avanza_tracker.py:114:        from portfolio.avanza_session import (
portfolio\avanza_tracker.py:115:            is_session_expiring_soon,
portfolio\avanza_tracker.py:116:            session_remaining_minutes,
portfolio\avanza_tracker.py:121:    remaining = session_remaining_minutes()
portfolio\avanza_tracker.py:123:        return "Avanza session not found. Run: python scripts/avanza_login.py"
portfolio\avanza_tracker.py:125:        return "Avanza session expired. Run: python scripts/avanza_login.py"
portfolio\avanza_tracker.py:126:    if is_session_expiring_soon(threshold_minutes=60.0):
portfolio\avanza_tracker.py:129:            f"Avanza session expires in {mins}min. "
portfolio\avanza_resilient_page.py:4:`portfolio/main.py` via `avanza_session.py`) open a headless Chromium at
portfolio\avanza_resilient_page.py:6:(OS sleep, memory pressure, WSL ping hiccup, external BankID re-auth) the
portfolio\avanza_resilient_page.py:52:    ``avanza_session.py`` which wants the same classifier without
portfolio\avanza\auth.py:1:"""Thread-safe TOTP authentication singleton for Avanza.
portfolio\avanza\auth.py:5:session regardless of how many threads call ``get_instance()``.
portfolio\avanza\auth.py:28:        credentials: Dict with keys ``username``, ``password``, ``totpSecret``.
portfolio\avanza\auth.py:46:    """Thread-safe singleton managing Avanza TOTP authentication.
portfolio\avanza\auth.py:50:        auth = AvanzaAuth.get_instance(username, password, totp_secret)
portfolio\avanza\auth.py:54:    on session expiry).
portfolio\avanza\auth.py:65:        authentication_session: str,
portfolio\avanza\auth.py:71:        self.authentication_session = authentication_session
portfolio\avanza\auth.py:79:        totp_secret: str,
portfolio\avanza\auth.py:84:        cost of TOTP authentication; subsequent callers return immediately.
portfolio\avanza\auth.py:97:                "totpSecret": totp_secret,
portfolio\avanza\auth.py:106:                authentication_session=getattr(client, "_authentication_session", ""),
scripts\fix_agent_dispatcher.py:192:    journal that surfaces critical errors to every Claude session would
scripts\fish_straddle.py:82:        from portfolio.avanza_session import place_buy_order
scripts\fish_straddle.py:94:        from portfolio.avanza_session import cancel_order as avanza_cancel
scripts\fish_straddle.py:104:        from portfolio.avanza_session import api_get
scripts\fish_straddle.py:122:        from portfolio.avanza_session import get_positions
scripts\fish_straddle.py:138:        from portfolio.avanza_session import get_quote, place_sell_order
scripts\fish_straddle.py:179:        from portfolio.avanza_session import get_buying_power, get_quote
scripts\fish_straddle.py:187:            log_msg('ABORT: get_buying_power() returned None — Avanza session may need refresh or API shape drift')
scripts\fish_straddle.py:303:                        from portfolio.avanza_session import get_quote as gq
scripts\fish_straddle.py:304:                        from portfolio.avanza_session import place_buy_order
scripts\fish_straddle.py:319:                        from portfolio.avanza_session import get_quote as gq
scripts\fish_straddle.py:320:                        from portfolio.avanza_session import place_buy_order
portfolio\avanza\client.py:44:                with ``"username"``, ``"password"``, and ``"totp_secret"`` when
portfolio\avanza\client.py:63:                totp_secret=avanza_cfg["totp_secret"],
portfolio\avanza\client.py:103:    def session(self) -> Any:
portfolio\avanza\client.py:105:        return self._auth.client._session
scripts\ft-health.sh:19:    local token chat_id
scripts\ft-health.sh:20:    token=$(python3 -c "import json; print(json.load(open('$CONFIG'))['telegram']['token'])")
scripts\ft-health.sh:21:    chat_id=$(python3 -c "import json; print(json.load(open('$CONFIG'))['telegram']['chat_id'])")
scripts\ft-health.sh:22:    curl -s -X POST "https://api.telegram.org/bot${token}/sendMessage" \
scripts\ft-health.sh:23:        -d chat_id="$chat_id" \
portfolio\daily_digest.py:274:            send_or_store(msg, config, category="daily_digest")
portfolio\digest.py:267:        send_or_store(msg, config, category="digest")
portfolio\crypto_scheduler.py:367:        send_or_store(msg, config, category="crypto_report")
portfolio\exit_optimizer.py:4:1. **Opportunity layer**: Monte Carlo path simulation for remaining-session
portfolio\exit_optimizer.py:9:   risk overrides (knock-out proximity, session end, volatility shock).
portfolio\exit_optimizer.py:12:instrument with price, volatility, and session data.
portfolio\exit_optimizer.py:16:    plan = compute_exit_plan(position, market, session_end, cost_model)
portfolio\exit_optimizer.py:83:        fill_prob: P(price reaches target before session end), 0.0-1.0.
portfolio\exit_optimizer.py:89:        quantile: Which quantile of session-max this candidate represents.
portfolio\exit_optimizer.py:109:        remaining_minutes: Minutes until session close.
portfolio\exit_optimizer.py:113:        session_max_distribution: Quantiles of the remaining-session max price.
portfolio\exit_optimizer.py:114:        session_min_distribution: Quantiles of the remaining-session min price.
portfolio\exit_optimizer.py:115:        stop_hit_prob: P(price drops to stop level before session end).
portfolio\exit_optimizer.py:124:    session_max_distribution: dict[str, float] = field(default_factory=dict)
portfolio\exit_optimizer.py:125:    session_min_distribution: dict[str, float] = field(default_factory=dict)
portfolio\exit_optimizer.py:157:            "session_max": self.session_max_distribution,
portfolio\exit_optimizer.py:158:            "session_min": self.session_min_distribution,
portfolio\exit_optimizer.py:205:        remaining_minutes: Minutes until session close.
portfolio\exit_optimizer.py:250:        Dict with session_max, session_min, terminal arrays and quantile dicts.
portfolio\exit_optimizer.py:252:    session_max = np.max(paths[:, 1:], axis=1)  # Exclude t=0
portfolio\exit_optimizer.py:253:    session_min = np.min(paths[:, 1:], axis=1)
portfolio\exit_optimizer.py:258:             for q, v in zip(quantiles, np.percentile(session_max, quantiles))}
portfolio\exit_optimizer.py:260:             for q, v in zip(quantiles, np.percentile(session_min, quantiles))}
portfolio\exit_optimizer.py:263:        "session_max": session_max,
portfolio\exit_optimizer.py:264:        "session_min": session_min,
portfolio\exit_optimizer.py:360:    session_max: np.ndarray | None = None,
portfolio\exit_optimizer.py:361:    session_min: np.ndarray | None = None,
portfolio\exit_optimizer.py:386:    # 4. Underlying session mismatch (warrant still trading but underlying closed)
portfolio\exit_optimizer.py:387:    # This would be detected by session_calendar, passed as a flag
portfolio\exit_optimizer.py:396:    if session_min is not None and position.financing_level:
portfolio\exit_optimizer.py:398:        p_knockout = float(np.mean(session_min <= stop_buffer))
portfolio\exit_optimizer.py:414:    session_min: np.ndarray | None = None,
portfolio\exit_optimizer.py:445:    if session_min is not None and position.financing_level:
portfolio\exit_optimizer.py:447:        p_knockout = float(np.mean(session_min <= stop_buffer))
portfolio\exit_optimizer.py:468:    session_end: datetime,
portfolio\exit_optimizer.py:479:    1. Simulates remaining-session price paths (Monte Carlo GBM)
portfolio\exit_optimizer.py:480:    2. Extracts session-max/min distributions
portfolio\exit_optimizer.py:481:    3. Generates candidate exits at quantile levels of session max
portfolio\exit_optimizer.py:489:        session_end: UTC datetime of session close.
portfolio\exit_optimizer.py:508:    if session_end.tzinfo is None:
portfolio\exit_optimizer.py:509:        session_end = session_end.replace(tzinfo=UTC)
portfolio\exit_optimizer.py:511:    remaining_min = max(0, (session_end - now).total_seconds() / 60)
portfolio\exit_optimizer.py:513:    # ---- Edge case: session over or almost over ----
portfolio\exit_optimizer.py:533:            provenance={"reason": "session_ended"},
portfolio\exit_optimizer.py:552:    session_max = stats["session_max"]
portfolio\exit_optimizer.py:553:    session_min = stats["session_min"]
portfolio\exit_optimizer.py:556:    # ---- 3. Generate candidate exits at session-max quantiles ----
portfolio\exit_optimizer.py:557:    target_prices = np.quantile(session_max, quantiles)
portfolio\exit_optimizer.py:572:        fill_prob = float(np.mean(session_max >= target))
portfolio\exit_optimizer.py:586:                                     session_max, session_min)
portfolio\exit_optimizer.py:632:                                              session_max, session_min)),
portfolio\exit_optimizer.py:642:        stop_prob = float(np.mean(session_min <= stop_price_usd))
portfolio\exit_optimizer.py:646:        stop_prob = float(np.mean(session_min <= stop_buffer))
portfolio\exit_optimizer.py:650:        candidates, position, market, remaining_min, session_min
portfolio\exit_optimizer.py:660:        session_max_distribution=stats["max_quantiles"],
portfolio\exit_optimizer.py:661:        session_min_distribution=stats["min_quantiles"],
portfolio\exit_optimizer.py:683:    session_end: datetime,
portfolio\exit_optimizer.py:699:        session_end: Session close time (UTC).
portfolio\exit_optimizer.py:752:    return compute_exit_plan(position, market, session_end, n_paths=n_paths)
dashboard\static\index_legacy.html:1978:          h += 'Session: $' + fn(tech.price.session_low, 2) + ' - $' + fn(tech.price.session_high, 2);
dashboard\static\index_legacy.html:2199:        var session = st.session_active !== false;
dashboard\static\index_legacy.html:2200:        var sessionColor = session ? "var(--grn)" : "var(--red)";
dashboard\static\index_legacy.html:2213:        h += '<div style="font-size:18px;font-weight:600;color:' + sessionColor + '">' + (session ? "OPEN" : "CLOSED") + '</div>';
portfolio\avanza\scanner.py:7:- TOTP (AvanzaClient) — preferred, faster
portfolio\avanza\scanner.py:8:- BankID session (avanza_session.api_get/api_post) — fallback
portfolio\avanza\scanner.py:38:# Dual-auth API helpers — try TOTP first, fall back to BankID session
portfolio\avanza\scanner.py:50:        - thread_safe: bool — True for TOTP (requests.Session), False for BankID (Playwright)
portfolio\avanza\scanner.py:52:    # Try TOTP client first (thread-safe, supports parallel fetching)
portfolio\avanza\scanner.py:68:        logger.debug("Scanner using TOTP client (thread-safe)")
portfolio\avanza\scanner.py:71:        logger.debug("TOTP client unavailable, falling back to BankID session")
portfolio\avanza\scanner.py:73:    # Fall back to BankID session (Playwright — NOT thread-safe, must be sequential)
portfolio\avanza\scanner.py:75:        from portfolio.avanza_session import api_get, api_post
portfolio\avanza\scanner.py:89:        logger.debug("Scanner using BankID session (sequential only)")
portfolio\avanza\scanner.py:93:            "No Avanza auth available. Either configure TOTP credentials "
portfolio\avanza\scanner.py:94:            "or run scripts/avanza_login.py for BankID session."
portfolio\avanza\scanner.py:309:        # TOTP: parallel fetch via thread pool
portfolio\avanza\scanner.py:317:        # BankID/Playwright: sequential (not thread-safe)
portfolio\elongir\state.py:79:    """Complete bot state, persisted to JSON between sessions."""
portfolio\elongir\runner.py:71:        return send_or_store(msg, config, category="elongir")
portfolio\elongir\runner.py:161:    """Main loop -- runs until killed or session close."""
portfolio\avanza\trading.py:70:    # 2026-04-17: match portfolio/avanza_session.py:590 convention — orders
portfolio\elongir\risk.py:110:def check_session(config: ElongirConfig) -> bool:
portfolio\elongir\risk.py:111:    """Check if current CET time is within the configured trading session.
portfolio\elongir\risk.py:113:    Returns True if within session hours.
portfolio\elongir\risk.py:132:    start_minutes = config.session_start_hour * 60 + config.session_start_minute
portfolio\elongir\risk.py:133:    end_minutes = config.session_end_hour * 60 + config.session_end_minute
portfolio\elongir\config.py:52:    session_start_hour: int = 8
portfolio\elongir\config.py:53:    session_start_minute: int = 30
portfolio\elongir\config.py:54:    session_end_hour: int = 21
portfolio\elongir\config.py:55:    session_end_minute: int = 30
portfolio\elongir\config.py:94:            session_start_hour=el.get("session_start_hour", 8),
portfolio\elongir\config.py:95:            session_start_minute=el.get("session_start_minute", 30),
portfolio\elongir\config.py:96:            session_end_hour=el.get("session_end_hour", 21),
portfolio\elongir\config.py:97:            session_end_minute=el.get("session_end_minute", 30),
portfolio\elongir\bot.py:4:1. Checks session window and daily limits
portfolio\elongir\bot.py:19:    check_session,
portfolio\elongir\bot.py:77:        if not check_session(self.cfg):
portfolio\elongir\bot.py:78:            logger.debug("Outside session -- skipping")
scripts\grid_fisher_probe.py:1:"""Dry-test the grid fisher pipeline against a live Avanza session.
scripts\grid_fisher_probe.py:46:    from portfolio import avanza_session
scripts\grid_fisher_probe.py:53:            session=avanza_session,
portfolio\fin_fish.py:263:def session_hours_remaining() -> float:
portfolio\fin_fish.py:264:    """Compute hours remaining in the Avanza warrant session (CET)."""
portfolio\fin_fish.py:1123:                        help="Override planning horizon (default: auto-compute from session).")
portfolio\fin_fish.py:1155:    hours = args.hours if args.hours > 0 else session_hours_remaining()
portfolio\fin_fish.py:1157:        # Outside trading hours — use next session (planning mode)
portfolio\fin_fish.py:1158:        hours = 13.67  # full session 08:15-21:55
portfolio\fin_fish.py:1359:            from portfolio.avanza_session import get_positions
scripts\golddigger_backtest_today.py:103:        in_session = (SESSION_START_H * 60 + SESSION_START_M) <= now_mins <= (SESSION_END_H * 60 + SESSION_END_M)
scripts\golddigger_backtest_today.py:111:            "valid": state.valid, "in_session": in_session,
scripts\golddigger_backtest_today.py:119:        if in_session and not in_position and signal.should_enter(state):
scripts\golddigger_backtest_today.py:185:        session_states = [s for s in states if s["in_session"] and s["valid"]]
scripts\golddigger_backtest_today.py:187:        if session_states:
scripts\golddigger_backtest_today.py:188:            smin = min(s["S"] for s in session_states)
scripts\golddigger_backtest_today.py:189:            smax = max(s["S"] for s in session_states)
scripts\golddigger_backtest_today.py:222:            session_states = [s for s in states if s["in_session"] and s["valid"]]
scripts\golddigger_backtest_today.py:224:            if session_states:
scripts\golddigger_backtest_today.py:225:                smin = min(s["S"] for s in session_states)
scripts\golddigger_backtest_today.py:226:                smax = max(s["S"] for s in session_states)
scripts\golddigger_backtest_today.py:254:        print("DETAILED TIMELINE (best config, session hours, every 5 min):")
scripts\golddigger_backtest_today.py:261:        session_states = [s for s in states if s["in_session"] and s["valid"]]
scripts\golddigger_backtest_today.py:265:        for s in session_states:
scripts\golddigger_backtest_today.py:283:        if session_states:
scripts\golddigger_backtest_today.py:284:            gold_high = max(s["gold"] for s in session_states)
scripts\golddigger_backtest_today.py:285:            gold_low = min(s["gold"] for s in session_states)
scripts\golddigger_backtest_today.py:297:        session_klines = []
scripts\golddigger_backtest_today.py:303:                session_klines.append((dt.strftime("%H:%M"), float(kl[4])))
scripts\golddigger_backtest_today.py:304:        if session_klines:
scripts\golddigger_backtest_today.py:305:            prices = [p for _, p in session_klines]
scripts\golddigger_backtest_today.py:306:            print(f"\n  Gold session: ${min(prices):.1f} - ${max(prices):.1f} ({(max(prices)-min(prices))/min(prices)*100:.2f}%)")
scripts\golddigger_backtest_today.py:310:            for t, p in session_klines:
portfolio\fin_snipe.py:15:from portfolio.avanza_session import api_get, verify_session
portfolio\fin_snipe.py:46:    """Return all Avanza stop-loss orders for the current session."""
portfolio\fin_snipe.py:260:    if not verify_session():
portfolio\fin_snipe.py:261:        raise SystemExit("Avanza session invalid or expired. Run scripts/avanza_login.py first.")
dashboard\static\js\render\trading-status-card.js:50:  const sessionPill = document.createElement("span");
dashboard\static\js\render\trading-status-card.js:51:  sessionPill.style.fontSize = "var(--ty-xs)";
dashboard\static\js\render\trading-status-card.js:52:  sessionPill.style.color = payload?.session_open ? "var(--grn)" : "var(--txm)";
dashboard\static\js\render\trading-status-card.js:53:  sessionPill.textContent = payload?.session_open ? "session open" : "session closed";
dashboard\static\js\render\trading-status-card.js:54:  titleRow.append(sessionPill);
portfolio\fish_instrument_finder.py:42:    """Search Avanza via the session API."""
portfolio\fish_instrument_finder.py:44:        from portfolio.avanza_session import api_post
portfolio\fish_instrument_finder.py:56:        from portfolio.avanza_session import get_quote
portfolio\fish_instrument_finder.py:66:        from portfolio.avanza_session import api_get
scripts\health_check.py:276:    if "nested session" in text.lower():
scripts\health_check.py:277:        return "fail", "Nested session errors"
scripts\health_check.py:535:        ok = send_or_store(msg, config, category="health")
scripts\iskbet.py:104:        token = config["telegram"]["token"]
scripts\iskbet.py:105:        chat_id = config["telegram"]["chat_id"]
scripts\iskbet.py:113:            f"https://api.telegram.org/bot{token}/sendMessage",
scripts\iskbet.py:114:            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
scripts\iskbet.py:205:    # Warn if replacing active session
scripts\iskbet.py:214:                        print(f"WARNING: Replacing active session (was scanning {', '.join(old.get('tickers', []))})")
scripts\loop_health_report.py:190:        if cfg.get("telegram", {}).get("token"):
scripts\loop_health_report.py:194:        print("(no telegram token in config — printing instead)")
portfolio\fin_snipe_manager.py:34:from portfolio.avanza_session import verify_session
portfolio\fin_snipe_manager.py:45:from portfolio.session_calendar import get_session_info
portfolio\fin_snipe_manager.py:84:    Categories: 'session_expired', 'naked_position', 'execution_failure',
portfolio\fin_snipe_manager.py:104:            send_or_store(message, config, category="error")
portfolio\fin_snipe_manager.py:117:def _new_session_id() -> str:
portfolio\fin_snipe_manager.py:161:    session_id: str | None = None,
portfolio\fin_snipe_manager.py:183:        if session_id is not None:
portfolio\fin_snipe_manager.py:184:            entry["session_id"] = session_id
portfolio\fin_snipe_manager.py:405:    session = get_session_info("warrant", underlying=snapshot.get("ticker"))
portfolio\fin_snipe_manager.py:411:        or not session.is_open
portfolio\fin_snipe_manager.py:412:        or session.remaining_minutes < 2
portfolio\fin_snipe_manager.py:487:            session.session_end,
portfolio\fin_snipe_manager.py:679:    session_id: str,
portfolio\fin_snipe_manager.py:689:        "session_id": session_id,
portfolio\fin_snipe_manager.py:691:        "cycle_id": f"{session_id}:{cycle_index}",
portfolio\fin_snipe_manager.py:705:    session_id: str,
portfolio\fin_snipe_manager.py:721:        session_id=session_id,
portfolio\fin_snipe_manager.py:786:    session_id: str,
portfolio\fin_snipe_manager.py:795:            "session_id": session_id,
portfolio\fin_snipe_manager.py:797:            "cycle_id": f"{session_id}:{cycle_index}",
portfolio\fin_snipe_manager.py:806:    session_id: str,
portfolio\fin_snipe_manager.py:818:            "session_id": session_id,
portfolio\fin_snipe_manager.py:820:            "cycle_id": f"{session_id}:{cycle_index}",
portfolio\fin_snipe_manager.py:836:    session_id: str,
portfolio\fin_snipe_manager.py:847:        session_id=session_id,
portfolio\fin_snipe_manager.py:1563:    session_id: str | None = None,
portfolio\fin_snipe_manager.py:1569:    session_id = session_id or _new_session_id()
portfolio\fin_snipe_manager.py:1575:        if not verify_session():
portfolio\fin_snipe_manager.py:1578:                    "session_expired",
portfolio\fin_snipe_manager.py:1579:                    "*SNIPE ALERT* Avanza session expired — cannot manage orders. Re-login required.",
portfolio\fin_snipe_manager.py:1581:            raise RuntimeError("Avanza session invalid or expired.")
portfolio\fin_snipe_manager.py:1595:            session_id=session_id,
portfolio\fin_snipe_manager.py:1615:                session_id=session_id,
portfolio\fin_snipe_manager.py:1650:            session_id=session_id,
portfolio\fin_snipe_manager.py:1662:            session_id=session_id,
portfolio\fin_snipe_manager.py:1685:    session_id = _new_session_id()
portfolio\fin_snipe_manager.py:1689:        owner=session_id,
portfolio\fin_snipe_manager.py:1709:            session_id=session_id,
scripts\loop_health_watchdog.py:112:      - config.json missing or has no telegram.token
scripts\loop_health_watchdog.py:124:        if not cfg.get("telegram", {}).get("token"):
portfolio\fx_rates.py:89:        send_or_store(msg, config, category="error")
portfolio\fish_monitor_smart.py:64:    """Signal-aware position monitor for fishing sessions.
portfolio\fish_monitor_smart.py:98:        self.session_high = entry_price
portfolio\fish_monitor_smart.py:99:        self.session_low = entry_price
portfolio\fish_monitor_smart.py:362:        - Time: 3h tighten, session end force sell
portfolio\fish_monitor_smart.py:517:            f"  Session: H ${self.session_high:.2f} / L ${self.session_low:.2f} | "
portfolio\fish_monitor_smart.py:622:            Stop after N checks (0 = run until session end or manual stop).
portfolio\fish_monitor_smart.py:629:            All exit signals triggered during the session.
portfolio\fish_monitor_smart.py:652:                    self.session_high = max(self.session_high, price)
portfolio\fish_monitor_smart.py:653:                    self.session_low = min(self.session_low, price)
portfolio\fish_monitor_smart.py:705:                    "session_high": self.session_high,
portfolio\fish_monitor_smart.py:706:                    "session_low": self.session_low,
portfolio\fish_monitor_smart.py:729:            "session_high": self.session_high,
portfolio\fish_monitor_smart.py:730:            "session_low": self.session_low,
portfolio\focus_analysis.py:235:    send_or_store(msg, config, category="analysis")
scripts\monitor_silver_exit.py:23:from portfolio.avanza_session import api_get, get_quote, get_positions
scripts\monitor_silver_exit.py:72:        send_or_store(msg, _CONFIG, category="analysis")
portfolio\golddigger\bot.py:4:1. Checks kill switch and session window
portfolio\golddigger\bot.py:88:    def _in_session(self, hour: int, minute: int) -> bool:
portfolio\golddigger\bot.py:89:        """Check if current Stockholm time is within trading session.
portfolio\golddigger\bot.py:91:        Includes flatten time (session_end) so we can still exit positions
portfolio\golddigger\bot.py:94:        start = self.cfg.session_start_hour * 60 + self.cfg.session_start_minute
portfolio\golddigger\bot.py:95:        end = self.cfg.session_end_hour * 60 + self.cfg.session_end_minute
portfolio\golddigger\bot.py:101:        flatten = self.cfg.session_end_hour * 60 + self.cfg.session_end_minute
portfolio\golddigger\bot.py:132:        # Outside session?
portfolio\golddigger\bot.py:133:        if not self._in_session(hour, minute):
portfolio\golddigger\bot.py:134:            logger.debug("Outside session (%02d:%02d) — skipping", hour, minute)
portfolio\golddigger\bot.py:186:        # --- Flatten at session end ---
portfolio\golddigger\config.py:46:    # whole pre-US window at the session level.
portfolio\golddigger\config.py:47:    session_start_hour: int = 8
portfolio\golddigger\config.py:48:    session_start_minute: int = 30
portfolio\golddigger\config.py:49:    session_end_hour: int = 21
portfolio\golddigger\config.py:50:    session_end_minute: int = 30
portfolio\golddigger\config.py:119:    session_check_interval: int = 300     # 5 min Avanza session health check
portfolio\golddigger\config.py:203:            session_check_interval=gd.get("session_check_interval", 300),
portfolio\golddigger\data_provider.py:4:FRED for US10Y yield, and Avanza Playwright session for certificate bid/ask.
portfolio\golddigger\state.py:27:    """Complete bot state, persisted to JSON between sessions."""
portfolio\golddigger\runner.py:3:Orchestrates the 30-second poll cycle, Playwright session management,
portfolio\golddigger\runner.py:20:    check_session_alive,
portfolio\golddigger\runner.py:93:        send_or_store(msg, config, category="golddigger")
portfolio\golddigger\runner.py:124:    """Initialize Playwright and load Avanza session.
portfolio\golddigger\runner.py:139:        # Navigate to Avanza to establish session
portfolio\golddigger\runner.py:141:        logger.info("Playwright session loaded")
portfolio\golddigger\runner.py:258:                logger.warning("No Playwright session — running without certificate prices")
portfolio\golddigger\runner.py:271:        _last_session_check = time.time()
portfolio\golddigger\runner.py:280:                _report.snapshot_collected = True  # GoldDigger always has data from session
portfolio\golddigger\runner.py:283:                if live and page and (time.time() - _last_session_check) > cfg.session_check_interval:
portfolio\golddigger\runner.py:284:                    session_ok = False
portfolio\golddigger\runner.py:286:                        if check_session_alive(page):
portfolio\golddigger\runner.py:287:                            session_ok = True
portfolio\golddigger\runner.py:293:                    if not session_ok:
portfolio\golddigger\runner.py:294:                        logger.error("Avanza session expired after 3 checks — "
portfolio\golddigger\runner.py:302:                        if check_session_alive(page):
portfolio\golddigger\runner.py:322:                            if page and check_session_alive(page):
portfolio\golddigger\runner.py:338:                    _last_session_check = time.time()
portfolio\golddigger\runner.py:339:                    _report.session_alive = session_ok
dashboard\static\js\views\settings.js:142:    // to /logout which sends Set-Cookie: pf_dashboard_token=; Max-Age=0
scripts\probe_oil_warrants.py:3:Opens a headless Playwright session against the existing
scripts\probe_oil_warrants.py:13:are independent so this won't conflict with the metals session.
scripts\probe_oil_warrants.py:46:        print("Run metals_loop or BankID re-auth first.", file=sys.stderr)
scripts\probe_oil_warrants.py:49:    print(f"Opening Avanza session via {STORAGE_STATE}")
scripts\probe_oil_warrants.py:55:            # Warm the session — visiting the home page validates cookies
scripts\pf.py:139:        token = config["telegram"]["token"]
scripts\pf.py:140:        chat_id = config["telegram"]["chat_id"]
scripts\pf.py:148:            f"https://api.telegram.org/bot{token}/sendMessage",
scripts\pf.py:149:            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
dashboard\static\js\views\golddigger.js:91:  // /api/golddigger normalizes to: composite_score, session_active,
dashboard\static\js\views\golddigger.js:93:  // drafts read s_t / session_open / gold_usd / usd_sek / confirms which
dashboard\static\js\views\golddigger.js:97:    _kpi("Session",  s.session_active ? "ACTIVE" : "CLOSED"),
scripts\setup_tunnel.bat:36:REM --- Safety gate: refuse to run if dashboard_token is empty ---
scripts\setup_tunnel.bat:40:REM _get_dashboard_token + check_token. Empty token returns 200 to all callers.
scripts\setup_tunnel.bat:43:"%REPO_ROOT%\.venv\Scripts\python.exe" -c "import json,sys; cfg=json.load(open('config.json',encoding='utf-8')); t=cfg.get('dashboard_token') or ''; sys.exit(0 if len(t)>=16 else 1)"
scripts\setup_tunnel.bat:45:    echo ERROR: dashboard_token in config.json is empty or too short.
scripts\setup_tunnel.bat:50:    echo        Then set it in config.json under "dashboard_token".
scripts\setup_tunnel.bat:55:echo [0/6] dashboard_token present — auth gate satisfied.
scripts\signal-research.bat:27:  "Set-Content -Path 'Q:\finance-analyzer\data\signal-research-progress.json' -Value ('{\"session_started\":\"%TS_START%\",\"current_phase\":\"starting\",\"status\":\"starting\",\"phases_completed\":[],\"notes\":\"Session launched, waiting for Claude to begin\"}')"
scripts\signal-research.bat:29:REM --- Log: session starting ---
scripts\signal-research.bat:33:echo [%TS_START%] Signal Research session starting...
portfolio\iskbets.py:39:    """Load per-session ISKBETS config. Returns dict or None if disabled/expired."""
portfolio\iskbets.py:84:    send_or_store(msg, config, category="iskbets")
portfolio\iskbets.py:90:    log_message(msg, category="iskbets", sent=False)
portfolio\iskbets.py:632:    session_cfg = _load_config()
portfolio\iskbets.py:633:    if not session_cfg:
portfolio\iskbets.py:670:        tickers = session_cfg.get("tickers", [])
portfolio\iskbets.py:873:        # Check if session is active
portfolio\iskbets.py:874:        session_cfg = _load_config()
portfolio\iskbets.py:875:        if session_cfg:
portfolio\iskbets.py:876:            tickers = session_cfg.get("tickers", [])
portfolio\http_retry.py:20:                     session=None):
portfolio\http_retry.py:25:    requester = session or requests
portfolio\http_retry.py:75:               label="", headers=None, params=None, timeout=30, session=None,
portfolio\http_retry.py:82:                            headers=headers, params=params, session=session)
portfolio\grid_fisher.py:77:    p_fill_session: Optional[float] = None
portfolio\grid_fisher.py:90:            "p_fill_session": self.p_fill_session,
portfolio\grid_fisher.py:105:            p_fill_session=d.get("p_fill_session"),
portfolio\grid_fisher.py:123:    session_pnl_sek: float = 0.0
portfolio\grid_fisher.py:124:    fills_this_session: int = 0
portfolio\grid_fisher.py:141:            "session_pnl_sek": self.session_pnl_sek,
portfolio\grid_fisher.py:142:            "fills_this_session": self.fills_this_session,
portfolio\grid_fisher.py:161:            session_pnl_sek=float(d.get("session_pnl_sek", 0.0) or 0.0),
portfolio\grid_fisher.py:162:            fills_this_session=int(d.get("fills_this_session", 0) or 0),
portfolio\grid_fisher.py:185:    def session_loss_breached(self) -> bool:
portfolio\grid_fisher.py:186:        return self.session_pnl_sek <= -abs(GRID_PER_SESSION_LOSS_LIMIT_SEK)
portfolio\grid_fisher.py:200:    session_id: str = ""
portfolio\grid_fisher.py:203:    global_session_pnl_sek: float = 0.0
portfolio\grid_fisher.py:210:            "session_id": self.session_id,
portfolio\grid_fisher.py:213:            "global_session_pnl_sek": self.global_session_pnl_sek,
portfolio\grid_fisher.py:224:            session_id=str(d.get("session_id", "")),
portfolio\grid_fisher.py:227:            global_session_pnl_sek=float(
portfolio\grid_fisher.py:228:                d.get("global_session_pnl_sek", 0.0) or 0.0
portfolio\grid_fisher.py:248:def _today_session_id() -> str:
portfolio\grid_fisher.py:340:        return GridFisherState(session_id=_today_session_id())
portfolio\grid_fisher.py:349:        return GridFisherState(session_id=_today_session_id())
portfolio\grid_fisher.py:358:        return GridFisherState(session_id=_today_session_id())
portfolio\grid_fisher.py:372:# carries AZAPERSISTENCE, csid, cstoken, AZACSRF — session-authoritative
portfolio\grid_fisher.py:431:def roll_session_if_new_day(state: GridFisherState) -> bool:
portfolio\grid_fisher.py:432:    """If the session date has advanced, reset per-session counters.
portfolio\grid_fisher.py:434:    Returns ``True`` if the session was rolled (so the caller can persist
portfolio\grid_fisher.py:439:    today = _today_session_id()
portfolio\grid_fisher.py:440:    if state.session_id == today:
portfolio\grid_fisher.py:443:        "grid_fisher: rolling session %s -> %s",
portfolio\grid_fisher.py:444:        state.session_id or "<new>",
portfolio\grid_fisher.py:447:    state.session_id = today
portfolio\grid_fisher.py:448:    state.global_session_pnl_sek = 0.0
portfolio\grid_fisher.py:451:        inst.session_pnl_sek = 0.0
portfolio\grid_fisher.py:452:        inst.fills_this_session = 0
portfolio\grid_fisher.py:506:    inst.fills_this_session += 1
portfolio\grid_fisher.py:522:        inst.session_pnl_sek += realised
portfolio\grid_fisher.py:572:    Currently uses session-wide P&L; richer drawdown tracking lives in
portfolio\grid_fisher.py:575:    # Per-session global loss limit = sum of per-instrument limits.
portfolio\grid_fisher.py:579:    if state.global_session_pnl_sek <= threshold:
portfolio\grid_fisher.py:580:        return f"global_session_pnl<{threshold:.0f}sek"
portfolio\grid_fisher.py:701:    """High-level driver — owns state + session + signal/quote callables.
portfolio\grid_fisher.py:703:    Constructor takes callables instead of importing the live session
portfolio\grid_fisher.py:705:    session that records calls instead of hitting Avanza.
portfolio\grid_fisher.py:709:      ``session.place_buy_order(orderbook_id, price, volume)``
portfolio\grid_fisher.py:710:      ``session.place_sell_order(orderbook_id, price, volume)``
portfolio\grid_fisher.py:711:      ``session.place_stop_loss(orderbook_id, trigger_price, sell_price, volume)``
portfolio\grid_fisher.py:712:      ``session.cancel_order(order_id)``
portfolio\grid_fisher.py:713:      ``session.get_open_orders()``
portfolio\grid_fisher.py:714:      ``session.get_positions()``
portfolio\grid_fisher.py:715:      ``session.get_quote(orderbook_id)``  -> dict with at least 'buy' (bid)
portfolio\grid_fisher.py:728:        session: Any,
portfolio\grid_fisher.py:752:        self.session = session
portfolio\grid_fisher.py:799:    def _safe_session_call(self, fn, *args, default=None, **kwargs):
portfolio\grid_fisher.py:800:        """Invoke an Avanza session method from a persistent worker thread.
portfolio\grid_fisher.py:803:        page; the REST avanza_session module that grid_fisher uses internally
portfolio\grid_fisher.py:808:        that but breaks the *next* call: the avanza_session module caches
portfolio\grid_fisher.py:813:        GridFisher — all session calls land on the same thread and the
portfolio\grid_fisher.py:823:        if getattr(self, "_session_executor", None) is None:
portfolio\grid_fisher.py:824:            self._session_executor = concurrent.futures.ThreadPoolExecutor(
portfolio\grid_fisher.py:825:                max_workers=1, thread_name_prefix="grid-fisher-session",
portfolio\grid_fisher.py:831:        future = self._session_executor.submit(_runner)
portfolio\grid_fisher.py:835:            self._log("session_call_timeout",
portfolio\grid_fisher.py:839:            self._log("session_call_error",
portfolio\grid_fisher.py:848:        executor = getattr(self, "_session_executor", None)
portfolio\grid_fisher.py:882:            result = self._safe_session_call(
portfolio\grid_fisher.py:883:                self.session.cancel_order, tier.order_id, default=None,
portfolio\grid_fisher.py:889:                          error="session_call returned None")
portfolio\grid_fisher.py:937:        if inst.session_loss_breached():
portfolio\grid_fisher.py:939:                      ticker=inst.ticker, session_pnl=inst.session_pnl_sek)
portfolio\grid_fisher.py:994:            result = self._safe_session_call(
portfolio\grid_fisher.py:995:                self.session.place_buy_order,
portfolio\grid_fisher.py:1003:                          error="session_call returned None")
portfolio\grid_fisher.py:1057:            result = self._safe_session_call(
portfolio\grid_fisher.py:1058:                self.session.place_sell_order,
portfolio\grid_fisher.py:1066:                          error="session_call returned None")
portfolio\grid_fisher.py:1086:        # errors (March 3 incident — see avanza_session.cancel_stop_loss
portfolio\grid_fisher.py:1090:            cancel_fn = getattr(self.session, "cancel_stop_loss", None)
portfolio\grid_fisher.py:1092:                self._safe_session_call(
portfolio\grid_fisher.py:1099:            result = self._safe_session_call(
portfolio\grid_fisher.py:1100:                self.session.place_stop_loss,
portfolio\grid_fisher.py:1109:                          error="session_call returned None")
portfolio\grid_fisher.py:1129:    # ---- session entry ----------------------------------------------------
portfolio\grid_fisher.py:1173:        if roll_session_if_new_day(self.state):
portfolio\grid_fisher.py:1174:            self._log("session_roll", session_id=self.state.session_id)
portfolio\grid_fisher.py:1177:        # Wrapping the read calls in _safe_session_call moves the sync
portfolio\grid_fisher.py:1185:        open_orders_raw = self._safe_session_call(
portfolio\grid_fisher.py:1186:            self.session.get_open_orders, default=None,
portfolio\grid_fisher.py:1188:        positions_raw = self._safe_session_call(
portfolio\grid_fisher.py:1189:            self.session.get_positions, default=None,
portfolio\grid_fisher.py:1226:        # ``should_halt_global`` actually sees the running session loss.
portfolio\grid_fisher.py:1228:        # ticks — instrument session_pnl_sek already aggregates every
portfolio\grid_fisher.py:1229:        # realised sell since the last roll_session.
portfolio\grid_fisher.py:1230:        self.state.global_session_pnl_sek = sum(
portfolio\grid_fisher.py:1231:            inst.session_pnl_sek
portfolio\grid_fisher.py:1318:            # quote against the live session when the caller didn't supply
portfolio\grid_fisher.py:1325:                quote = self._safe_session_call(
portfolio\grid_fisher.py:1326:                    self.session.get_quote, ob_id, default=None,
portfolio\grid_fisher.py:1331:                              error="session_call returned None")
portfolio\grid_fisher.py:1398:                    self._safe_session_call(
portfolio\grid_fisher.py:1399:                        self.session.cancel_order, tier.order_id,
portfolio\grid_fisher.py:1406:                    self.session, "cancel_stop_loss", None,
portfolio\grid_fisher.py:1409:                    self._safe_session_call(
portfolio\grid_fisher.py:1415:            quote = self._safe_session_call(
portfolio\grid_fisher.py:1416:                self.session.get_quote, inst.ob_id, default=None,
portfolio\grid_fisher.py:1425:            result = self._safe_session_call(
portfolio\grid_fisher.py:1426:                self.session.place_sell_order,
portfolio\grid_fisher.py:1433:                          error="session_call returned None")
portfolio\grid_fisher.py:1463:        "session_id": state.session_id,
portfolio\grid_fisher.py:1466:        "global_session_pnl_sek": round(state.global_session_pnl_sek, 2),
portfolio\grid_fisher.py:1476:                "session_pnl_sek": round(inst.session_pnl_sek, 2),
portfolio\grid_fisher.py:1477:                "fills_this_session": inst.fills_this_session,
scripts\sync_dashboard.py:98:    """Read dashboard_token from config.json (if configured)."""
scripts\sync_dashboard.py:101:        return cfg.get("dashboard_token") or None
portfolio\health.py:52:def reset_session_start():
portfolio\health.py:56:    from a previous session's health_state.json.
scripts\verify_tunnel.py:41:COOKIE_NAME = "pf_dashboard_token"
scripts\verify_tunnel.py:47:    return cfg.get("dashboard_token") or ""
scripts\verify_tunnel.py:63:        print("FATAL: dashboard_token missing or empty in config.json")
portfolio\grid_tiers.py:7:Avanza session.
portfolio\grid_fisher_config.py:70:# placements until the next session.
scripts\verify_tunnel_alerted.py:27:  - exit 2: config.json's dashboard_token missing — broken deployment
scripts\verify_tunnel_alerted.py:43:def send_telegram_direct(msg: str, token: str, chat_id: str) -> bool:
scripts\verify_tunnel_alerted.py:51:    api = f"https://api.telegram.org/bot{token}/sendMessage"
scripts\verify_tunnel_alerted.py:55:            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
scripts\verify_tunnel_alerted.py:64:                json={"chat_id": chat_id, "text": msg},
scripts\verify_tunnel_alerted.py:95:    chat_id = tg.get("chat_id", "")
scripts\verify_tunnel_alerted.py:99:    if token and chat_id:
scripts\verify_tunnel_alerted.py:100:        sent = send_telegram_direct(msg, token, chat_id)
scripts\verify_tunnel_alerted.py:109:                "telegram_configured": bool(token and chat_id),
scripts\write_research_outputs.py:1:"""Write Phase 1-5 research outputs for signal research session 2026-04-26."""
portfolio\llama_server.py:145:            # CSV format: "name.exe","PID","session","session#","mem"
portfolio\llm_batch.py:19:reduction in fingpt inference time. See project_fingpt_llmbatch_session
portfolio\message_throttle.py:105:        send_or_store(text, config, category="analysis")
portfolio\message_store.py:87:def log_message(text, category="analysis", sent=False):
portfolio\message_store.py:117:    token = config.get("telegram", {}).get("token")
portfolio\message_store.py:118:    chat_id = config.get("telegram", {}).get("chat_id")
portfolio\message_store.py:119:    if not token or not chat_id:
portfolio\message_store.py:120:        logger.warning("Telegram token/chat_id not configured")
portfolio\message_store.py:138:        f"https://api.telegram.org/bot{token}/sendMessage",
portfolio\message_store.py:140:        json_body={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
portfolio\message_store.py:161:                f"https://api.telegram.org/bot{token}/sendMessage",
portfolio\message_store.py:163:                json_body={"chat_id": chat_id, "text": msg},
portfolio\message_store.py:170:def send_or_store(msg, config, category="analysis"):
portfolio\message_store.py:181:        config: Full config dict (needs ``telegram.token`` and ``telegram.chat_id``).
portfolio\message_store.py:196:        log_message(cleaned, category=category, sent=False)
portfolio\message_store.py:204:            log_message(cleaned, category=category, sent=False)
portfolio\message_store.py:210:        log_message(cleaned, category=category, sent=sent_ok)
portfolio\message_store.py:217:        log_message(cleaned, category=category, sent=False)
portfolio\loop_contract.py:5:trigger a self-healing Claude Code session.
portfolio\loop_contract.py:112:# (heavy LLM batch + busy news_event + Avanza session re-auth) — not hangs.
portfolio\loop_contract.py:129:SELF_HEAL_COOLDOWN_S = 1800  # 30 minutes between sessions
portfolio\loop_contract.py:154:# Telegram delivery, and journal flush — enough that a healthy slow session
portfolio\loop_contract.py:785:    while a separate WARNING gets logged so the next session sees what
portfolio\loop_contract.py:1532:    session_alive: bool = True                # Avanza session health
portfolio\loop_contract.py:1575:    if not report.session_alive:
portfolio\loop_contract.py:1577:            invariant="session_alive",
portfolio\loop_contract.py:1579:            message="Avanza session is dead. Trading disabled until session renewed.",
portfolio\loop_contract.py:1642:    session_alive: bool = True                 # Avanza session (GoldDigger only)
portfolio\loop_contract.py:1669:    # 2. Snapshot collected (Elongir) / session alive (GoldDigger)
portfolio\loop_contract.py:1679:    if not report.session_alive:
portfolio\loop_contract.py:1681:            invariant="session_alive",
portfolio\loop_contract.py:1683:            message=f"{name}: Avanza session is dead",
portfolio\loop_contract.py:1791:        """Check if enough time has passed since last self-healing session."""
portfolio\loop_contract.py:1795:        """Record that a self-healing session was triggered."""
portfolio\loop_contract.py:1831:    """Build a diagnostic prompt for the self-healing Claude Code session."""
portfolio\loop_contract.py:2234:        # failure (missing token/chat_id, non-OK sendMessage response).
portfolio\loop_contract.py:2238:        sent_ok = send_or_store(msg, config, category=CONTRACT_ALERT_CATEGORY)
portfolio\loop_contract.py:2263:    """Spawn a Claude Code session to diagnose and fix critical violations."""
portfolio\loop_contract.py:2279:            "Triggering self-healing session for %d critical violation(s) [%s]",
portfolio\main.py:772:            send_or_store(_fail_msg, config, category="error")
portfolio\main.py:936:                    send_or_store(msg, config, category="error")
portfolio\main.py:948:                    send_or_store(msg, config, category="error")
portfolio\main.py:1021:                send_or_store(text, config, category="error")
portfolio\main.py:1033:        send_or_store(text, config, category="error")
portfolio\main.py:1156:                    send_or_store(msg, config, category="error")
portfolio\main.py:1162:    # Reset session start_time so uptime_seconds is accurate for this session
portfolio\main.py:1163:    from portfolio.health import reset_session_start
portfolio\main.py:1164:    reset_session_start()
scripts\win\crypto-loop.bat:9:REM Clear Claude Code session markers so any subagent invocation can launch.
scripts\win\install-mstr-loop-task.ps1:26:# MSTR is US-listed — only relevant during US session (15:30-22:00 CET).
scripts\win\install-mstr-loop-task.ps1:27:# The loop itself early-exits outside session, but starting earlier means
scripts\win\install-oil-loop-task.ps1:45:Write-Host "Run a one-shot probe with a live Avanza session to fill the catalog:"
scripts\win\install-rc-watchdog-task.ps1:3:# The watchdog proactively recycles RC servers before the 24h session timeout
scripts\win\pf-agent.bat:8:REM Clear Claude Code session markers — prevents "nested session" error when launched from
scripts\win\metals-loop.bat:7:REM Clear Claude Code session markers so Layer 2 agent can launch
scripts\win\pf-loop.bat:9:REM Clear Claude Code session markers so Layer 2 agent can launch
portfolio\session_calendar.py:1:"""Session calendar — instrument-specific trading hours and session state.
portfolio\session_calendar.py:3:Provides remaining-session time, session boundaries, and session mismatch
portfolio\session_calendar.py:7:    from portfolio.session_calendar import get_session_info
portfolio\session_calendar.py:8:    info = get_session_info("warrant", underlying="XAG-USD")
portfolio\session_calendar.py:9:    # info.remaining_minutes, info.session_end, info.is_extended, ...
portfolio\session_calendar.py:22:    """Trading session state for an instrument.
portfolio\session_calendar.py:25:        session_end: Absolute datetime (UTC) of normal session close.
portfolio\session_calendar.py:26:        extended_end: Absolute datetime (UTC) of extended session close, if applicable.
portfolio\session_calendar.py:29:        is_extended: Whether we're in the extended (evening) session.
portfolio\session_calendar.py:33:    session_end: datetime
portfolio\session_calendar.py:48:# We handle DST for EU sessions too.
portfolio\session_calendar.py:82:def _make_session_end(now: datetime, cet_hour: int, cet_minute: int) -> datetime:
portfolio\session_calendar.py:83:    """Create a UTC datetime for today's session end from CET time."""
portfolio\session_calendar.py:115:def get_session_info(instrument_type: str,
portfolio\session_calendar.py:118:    """Get current session state for an instrument.
portfolio\session_calendar.py:126:        SessionInfo with remaining time, phase, and session boundaries.
portfolio\session_calendar.py:135:        # Use midnight as "session end" — effectively infinite session
portfolio\session_calendar.py:140:            session_end=end,
portfolio\session_calendar.py:155:        session_end = now.replace(hour=close_utc, minute=0, second=0, microsecond=0)
portfolio\session_calendar.py:157:                   now.replace(hour=open_utc, minute=30, second=0) <= now < session_end)
portfolio\session_calendar.py:159:        remaining = max(0, (session_end - now).total_seconds() / 60) if is_open else 0
portfolio\session_calendar.py:166:            session_end=session_end,
portfolio\session_calendar.py:180:    session_end = _make_session_end(now, ch, cm)
portfolio\session_calendar.py:181:    session_open = _make_session_end(now, oh, om)
portfolio\session_calendar.py:184:    is_open = is_weekday and session_open <= now < session_end
portfolio\session_calendar.py:186:    remaining = max(0, (session_end - now).total_seconds() / 60) if is_open else 0
portfolio\session_calendar.py:193:        us_info = get_session_info("stock_us", now=now)
portfolio\session_calendar.py:197:        session_end=session_end,
portfolio\session_calendar.py:207:def remaining_session_minutes(instrument_type: str = "warrant",
portfolio\session_calendar.py:209:    """Shortcut: get remaining minutes for an instrument's session."""
portfolio\session_calendar.py:210:    info = get_session_info(instrument_type, now=now)
scripts\win\rc-keepalive.ps1:1:# rc-keepalive.ps1 - Soft-recycle idle RC servers to prevent Anthropic session delisting
scripts\win\rc-keepalive.ps1:5:# the session. The CLI keeps polling (logs stay fresh), but the session becomes invisible
scripts\win\rc-keepalive.ps1:15:# Servers with active work (>1 child session) are NEVER touched.
scripts\win\rc-keepalive.ps1:49:        $token  = $cfg.telegram.token
scripts\win\rc-keepalive.ps1:50:        $chatId = $cfg.telegram.chat_id
scripts\win\rc-keepalive.ps1:52:        $body = @{ chat_id = $chatId; text = $msg; parse_mode = "Markdown" } | ConvertTo-Json
scripts\win\rc-keepalive.ps1:53:        Invoke-RestMethod -Uri "https://api.telegram.org/bot$token/sendMessage" `
scripts\win\rc-keepalive.ps1:68:$childProcs = $allClaude | Where-Object { $_.CommandLine -match 'session-id' }
scripts\win\rc-keepalive.ps1:87:    # Active sessions are NEVER touched, even on wake
scripts\win\rc-keepalive.ps1:89:        Log "$name`: ${ageMin}m old, $childCount child sessions [ACTIVE WORK]. Protected."
scripts\win\rc-keepalive.ps1:114:    Log "$name`: recycling [$reason]. $childCount child sessions."
portfolio\multi_agent_layer2.py:148:    # this, when multi_agent=true fires, three specialist Claude sessions
portfolio\sentiment_shadow_backfill.py:6:session; every LLM signal's vote lands there with a clean schema. It's
portfolio\signal_decay_alert.py:8:Added 2026-04-30 after-hours research session. Prevents silent accuracy
portfolio\signal_decay_alert.py:9:erosion between manual audit sessions.
scripts\win\rc-server-ensure.ps1:7:#      process is alive and has a TCP socket open. This catches the case where a session
scripts\win\rc-server-ensure.ps1:12:# Design: sessions should live as long as the PC is on. Recycling kills context and can
scripts\win\rc-server-ensure.ps1:13:# leave spawned scripts/agents running unsupervised. Only recycle truly dead sessions.
scripts\win\rc-server-ensure.ps1:44:        $token  = $cfg.telegram.token
scripts\win\rc-server-ensure.ps1:45:        $chatId = $cfg.telegram.chat_id
scripts\win\rc-server-ensure.ps1:47:        $body = @{ chat_id = $chatId; text = $msg; parse_mode = "Markdown" } | ConvertTo-Json
scripts\win\rc-server-ensure.ps1:48:        Invoke-RestMethod -Uri "https://api.telegram.org/bot$token/sendMessage" `
scripts\win\rc-server.bat:7:REM Clear Claude Code session markers (prevents nested-session errors)
portfolio\telegram_notifications.py:52:    token = config["telegram"]["token"]
portfolio\telegram_notifications.py:53:    chat_id = config["telegram"]["chat_id"]
portfolio\telegram_notifications.py:55:        f"https://api.telegram.org/bot{token}/sendMessage",
portfolio\telegram_notifications.py:57:        json_body={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
portfolio\telegram_notifications.py:75:                f"https://api.telegram.org/bot{token}/sendMessage",
portfolio\telegram_notifications.py:77:                json_body={"chat_id": chat_id, "text": msg},
portfolio\telegram_notifications.py:139:        send_or_store(msg, config, category="analysis")
portfolio\reporting.py:685:        from portfolio.session_calendar import get_session_info
portfolio\reporting.py:697:                sess = get_session_info("warrant", underlying=underlying)
portfolio\reporting.py:717:                    position, market, sess.session_end,
portfolio\regime_alerts.py:181:        send_or_store(msg, config, category="regime")
portfolio\risk_management.py:402:    For each position, simulates remaining-session price paths and estimates
portfolio\risk_management.py:413:            - stop_hit_prob: P(hitting stop this session), 0.0-1.0
portfolio\risk_management.py:420:        from portfolio.session_calendar import remaining_session_minutes
portfolio\risk_management.py:422:        logger.warning("exit_optimizer or session_calendar not available")
portfolio\risk_management.py:448:        # Determine instrument type for session lookup
portfolio\risk_management.py:456:        # Get remaining session minutes
portfolio\risk_management.py:457:        remaining = remaining_session_minutes(inst_type)
portfolio\risk_management.py:481:        session_min = np.min(paths[:, 1:], axis=1)
portfolio\risk_management.py:482:        stop_hit_prob = float(np.mean(session_min <= stop_price))
portfolio\telegram_poller.py:45:        config: full app config dict (with telegram.token, telegram.chat_id)
portfolio\telegram_poller.py:48:        self.token = config["telegram"]["token"]
portfolio\telegram_poller.py:49:        self.chat_id = str(config["telegram"]["chat_id"])
portfolio\telegram_poller.py:130:            f"https://api.telegram.org/bot{self.token}/getUpdates",
portfolio\telegram_poller.py:174:        # Only process messages from our chat_id. Drop others without logging —
portfolio\telegram_poller.py:180:            if str(chat.get("id")) != self.chat_id:
portfolio\telegram_poller.py:238:            # Dispatch can raise (Avanza session, volume math, network) — we
portfolio\telegram_poller.py:375:                f"https://api.telegram.org/bot{self.token}/sendMessage",
portfolio\telegram_poller.py:378:                    "chat_id": self.chat_id,
portfolio\signals\calendar_seasonal.py:195:    ``_day_of_week_effect``.  Only true pre-holiday sessions (the
portfolio\short_horizon.py:24:    # Asian session (1-6 UTC) — 48-50%, slightly below baseline
portfolio\tickers.py:158:# 2026-04-11 research session changes:
portfolio\mstr_loop\config.py:6:- PHASE = "live"   — real orders on Avanza via BankID session
scripts\win\rc-watchdog.ps1:3:#   1. Proactive recycle: kill RC servers older than $MaxAgeHours so sessions
scripts\win\rc-watchdog.ps1:32:        $token  = $cfg.telegram.token
scripts\win\rc-watchdog.ps1:33:        $chatId = $cfg.telegram.chat_id
scripts\win\rc-watchdog.ps1:35:            Log "Telegram: missing token or chat_id in config"
scripts\win\rc-watchdog.ps1:38:        $body = @{ chat_id = $chatId; text = $msg; parse_mode = "Markdown" } | ConvertTo-Json
scripts\win\rc-watchdog.ps1:39:        Invoke-RestMethod -Uri "https://api.telegram.org/bot$token/sendMessage" `
portfolio\mstr_loop\execution.py:5:- live:   decisions place real Avanza orders via portfolio.avanza_session.
portfolio\mstr_loop\execution.py:12:  cleanly even if avanza_session/avanza_control are unavailable.
portfolio\mstr_loop\execution.py:13:- Live path lazy-imports avanza_session so shadow/paper users don't pay
portfolio\mstr_loop\execution.py:71:      2. Fall back to synthetic 100.0 SEK if Avanza session unavailable
portfolio\mstr_loop\execution.py:79:            from portfolio.avanza_session import get_quote
portfolio\mstr_loop\execution.py:176:    # keeps the Position in-memory only; the session.py EOD-flatten or
portfolio\mstr_loop\execution.py:439:        from portfolio.avanza_session import get_quote
portfolio\mstr_loop\execution.py:465:        from portfolio.avanza_session import get_quote
portfolio\mstr_loop\execution.py:467:        logger.exception("execution: avanza_session unavailable in live mode")
portfolio\mstr_loop\execution.py:480:        from portfolio.avanza_session import get_quote
portfolio\mstr_loop\execution.py:482:        logger.exception("execution: avanza_session unavailable in live mode")
portfolio\mstr_loop\execution.py:494:        from portfolio.avanza_session import place_buy_order
portfolio\mstr_loop\execution.py:496:        logger.exception("execution: avanza_session unavailable in live mode")
portfolio\mstr_loop\execution.py:511:        from portfolio.avanza_session import place_sell_order
portfolio\mstr_loop\execution.py:513:        logger.exception("execution: avanza_session unavailable in live mode")
portfolio\mstr_loop\risk.py:47:    Call once per cycle before gate checks. Rolls `session_start_equity`
portfolio\mstr_loop\risk.py:49:    session"); rolls `week_start_equity` forward when a new ISO week begins.
portfolio\mstr_loop\risk.py:59:    if state.session_start_ts != today_iso:
portfolio\mstr_loop\risk.py:60:        state.session_start_ts = today_iso
portfolio\mstr_loop\risk.py:61:        state.session_start_equity_sek = eq
portfolio\mstr_loop\risk.py:93:    if state.session_start_equity_sek > 0:
portfolio\mstr_loop\risk.py:94:        daily_pnl_pct = (current_equity - state.session_start_equity_sek) / state.session_start_equity_sek * 100
portfolio\mstr_loop\risk.py:180:    In quiet sessions atr_pct is small → tighter trail. In wild sessions
portfolio\mstr_loop\session.py:21:    for session-window gating since the 15-min EOD buffer absorbs slop).
portfolio\mstr_loop\session.py:54:def in_session_window(now: datetime.datetime | None = None) -> bool:
portfolio\mstr_loop\session.py:91:def seconds_until_next_session(now: datetime.datetime | None = None) -> int:
portfolio\mstr_loop\session.py:92:    """Seconds until the next session open (for the runner's idle sleep)."""
portfolio\mstr_loop\loop.py:4:  1. Check kill-switch + session window.
portfolio\mstr_loop\loop.py:19:from portfolio.mstr_loop import config, execution, risk, session, state, telegram_report
portfolio\mstr_loop\loop.py:76:    if session.kill_switch_active():
portfolio\mstr_loop\loop.py:102:    in_window = session.in_session_window()
portfolio\mstr_loop\loop.py:103:    if not in_window and not session.in_eod_flatten_window():
portfolio\mstr_loop\loop.py:104:        _log_poll(None, "outside_session_window", cycle_count)
portfolio\signals\forecast.py:60:# The correct structure (this session):
portfolio\strategies\golddigger_strategy.py:5:by the metals loop's Playwright session.
portfolio\mstr_loop\state.py:65:    session_started_ts: str = ""
portfolio\mstr_loop\state.py:69:    session_start_equity_sek: float = 0.0  # equity at the start of the current US session (for daily %)
portfolio\mstr_loop\state.py:71:    session_start_ts: str = ""             # resets daily at session-open
portfolio\mstr_loop\state.py:114:        session_started_ts=datetime.datetime.now(datetime.UTC).isoformat(),
portfolio\mstr_loop\state.py:136:        "session_started_ts": state.session_started_ts,
portfolio\mstr_loop\state.py:139:        "session_start_equity_sek": state.session_start_equity_sek,
portfolio\mstr_loop\state.py:141:        "session_start_ts": state.session_start_ts,
portfolio\mstr_loop\state.py:164:        session_started_ts=str(d.get("session_started_ts") or ""),
portfolio\mstr_loop\state.py:167:        session_start_equity_sek=float(d.get("session_start_equity_sek") or 0.0),
portfolio\mstr_loop\state.py:169:        session_start_ts=str(d.get("session_start_ts") or ""),
portfolio\mstr_loop\telegram_report.py:144:        send_or_store(text, cfg, category=category)
portfolio\mstr_loop\strategies\mean_reversion.py:121:        from portfolio.mstr_loop import session
portfolio\mstr_loop\strategies\mean_reversion.py:122:        if session.in_eod_flatten_window():
portfolio\mstr_loop\strategies\momentum_rider.py:125:        # Distance is ATR-adaptive when enabled so quiet sessions use a
portfolio\mstr_loop\strategies\momentum_rider.py:126:        # tighter trail and wild sessions a wider one.
portfolio\mstr_loop\strategies\momentum_rider.py:145:        from portfolio.mstr_loop import session
portfolio\mstr_loop\strategies\momentum_rider.py:146:        if session.in_eod_flatten_window():
portfolio\mstr_loop\strategies\overnight_gap.py:19:  1. Add is_monday_open() session helper (first 30min of Monday session).
portfolio\weekly_digest.py:272:    token = config.get("telegram", {}).get("token")
portfolio\weekly_digest.py:273:    chat_id = config.get("telegram", {}).get("chat_id")
portfolio\weekly_digest.py:275:    if not token or not chat_id:
portfolio\weekly_digest.py:276:        print("ERROR: Telegram config missing token or chat_id")
portfolio\signals\volume_flow.py:6:    2. VWAP Cross (price vs session VWAP)
portfolio\signals\volume_flow.py:64:    RESET at session boundaries (00:00 UTC).
portfolio\signals\volume_flow.py:67:    cumulated over the entire dataframe with no session reset, which made
portfolio\signals\volume_flow.py:69:    each day. A pure session VWAP must reset every UTC midnight so the
portfolio\signals\volume_flow.py:72:    Crypto/metals trade 24/7 with the convention that the daily session
portfolio\signals\volume_flow.py:84:    # Build a per-bar session id from the index date. groupby+cumsum gives
portfolio\signals\volume_flow.py:85:    # us cumulative-within-session sums in a single vectorized pass.
portfolio\signals\volume_flow.py:88:        # tz-aware: normalize to UTC so the session boundary is consistent.
portfolio\signals\volume_flow.py:91:            session_id = idx.tz_convert("UTC").date
portfolio\signals\volume_flow.py:93:            session_id = idx.date
portfolio\signals\volume_flow.py:94:        session_id = pd.Index(session_id)
portfolio\signals\volume_flow.py:95:        cum_vol = volume.groupby(session_id).cumsum()
portfolio\signals\volume_flow.py:96:        cum_vp = vp.groupby(session_id).cumsum()
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n -C 3 "claude|codex|remote-control|CLAUDECODE|CLAUDE_CODE_ENTRYPOINT" scripts/win' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 4 \"New-ScheduledTaskAction|schtasks /Create|cmd\\.exe|/c |CLAUDECODE|CLAUDE_CODE_ENTRYPOINT|ExecutionPolicy|ErrorActionPreference|Get-Content .*config|ConvertFrom-Json|Invoke-RestMethod\" scripts/win" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "set CLAUDECODE=|set CLAUDE_CODE_ENTRYPOINT=|claude |claude.exe|remote-control" scripts/win/*.bat scripts/win/*.ps1 scripts/*.bat' in Q:\finance-analyzer
 succeeded in 736ms:
scripts/win\crypto-loop.bat-7-
scripts/win\crypto-loop.bat-8-:restart
scripts/win\crypto-loop.bat-9-REM Clear Claude Code session markers so any subagent invocation can launch.
scripts/win\crypto-loop.bat:10:set CLAUDECODE=
scripts/win\crypto-loop.bat:11:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\crypto-loop.bat-12-echo [%date% %time%] Starting crypto loop...
scripts/win\crypto-loop.bat-13-.venv\Scripts\python.exe -u data\crypto_loop.py --loop > data\crypto_loop_out.txt 2>&1
scripts/win\crypto-loop.bat-14-set EXIT_CODE=%ERRORLEVEL%
--
scripts/win\adversarial-review.bat-1-@echo off
scripts/win\adversarial-review.bat-2-REM PF-AdversarialReview — Daily dual adversarial review (Codex + Claude)
scripts/win\adversarial-review.bat:3:REM Runs claude code CLI with the full review prompt.
scripts/win\adversarial-review.bat-4-REM Output: data\adversarial_review_out.txt
scripts/win\adversarial-review.bat-5-cd /d Q:\finance-analyzer
scripts/win\adversarial-review.bat:6:set CLAUDECODE=
scripts/win\adversarial-review.bat:7:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\adversarial-review.bat-8-
scripts/win\adversarial-review.bat-9-echo [%date% %time%] Starting adversarial review... >> data\adversarial_review_out.txt 2>&1
scripts/win\adversarial-review.bat-10-
scripts/win\adversarial-review.bat:11:claude -p "Follow /fgl protocol. Run a full dual adversarial review of the finance-analyzer codebase: partition into 8 subsystems (signals-core, orchestration, portfolio-risk, metals-core, avanza-api, signals-modules, data-external, infrastructure), create worktree with empty-baseline branches, run /codex:adversarial-review in background for each subsystem, write your own independent adversarial review, collect codex results, cross-critique in both directions, write synthesis doc. Commit all docs to main, push via Windows git, clean up worktrees. Do NOT ask for approval — follow /fgl rules. Spend your entire context on this." --allowedTools "Edit,Read,Bash,Write,Glob,Grep" --max-turns 80 >> data\adversarial_review_out.txt 2>&1
scripts/win\adversarial-review.bat-12-
scripts/win\adversarial-review.bat-13-echo [%date% %time%] Adversarial review finished (code %ERRORLEVEL%). >> data\adversarial_review_out.txt 2>&1
--
scripts/win\install-rc-keepalive-task.ps1-3-#
scripts/win\install-rc-keepalive-task.ps1-4-# Anthropic's server-side TTL is ~20 min without real user activity.
scripts/win\install-rc-keepalive-task.ps1-5-# Keepalive recycles idle servers at staggered thresholds (13/15/17 min)
scripts/win\install-rc-keepalive-task.ps1:6:# to keep them visible in the claude.ai/code picker.
scripts/win\install-rc-keepalive-task.ps1-7-#
scripts/win\install-rc-keepalive-task.ps1-8-# Also creates a wake-from-sleep trigger that runs keepalive in -Wake mode,
scripts/win\install-rc-keepalive-task.ps1-9-# which immediately recycles all idle servers (sleep guarantees staleness).
--
scripts/win\meta-learner-retrain.bat-1-@echo off
scripts/win\meta-learner-retrain.bat-2-REM PF-MetaLearnerRetrain — Daily LightGBM meta-learner retraining (low priority)
scripts/win\meta-learner-retrain.bat-3-cd /d Q:\finance-analyzer
scripts/win\meta-learner-retrain.bat:4:set CLAUDECODE=
scripts/win\meta-learner-retrain.bat:5:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\meta-learner-retrain.bat-6-set PYTHONPATH=Q:\finance-analyzer
scripts/win\meta-learner-retrain.bat-7-
scripts/win\meta-learner-retrain.bat-8-echo [%date% %time%] Starting meta-learner retraining... >> data\meta_learner_retrain_out.txt 2>&1
--
scripts/win\install-signal-research-task.ps1-36-    -Action $action `
scripts/win\install-signal-research-task.ps1-37-    -Trigger $trigger `
scripts/win\install-signal-research-task.ps1-38-    -Settings $settings `
scripts/win\install-signal-research-task.ps1:39:    -Description "Daily AI signal research: academic papers, web search, scoring, implementation, backtest, codex review. Runs Claude Opus." `
scripts/win\install-signal-research-task.ps1-40-    -RunLevel Highest
scripts/win\install-signal-research-task.ps1-41-
scripts/win\install-signal-research-task.ps1-42-Write-Host ""
--
scripts/win\metals-loop.bat-5-
scripts/win\metals-loop.bat-6-:restart
scripts/win\metals-loop.bat-7-REM Clear Claude Code session markers so Layer 2 agent can launch
scripts/win\metals-loop.bat:8:set CLAUDECODE=
scripts/win\metals-loop.bat:9:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\metals-loop.bat-10-echo [%date% %time%] Starting metals loop...
scripts/win\metals-loop.bat-11-.venv\Scripts\python.exe -u data\metals_loop.py > data\metals_loop_out.txt 2>&1
scripts/win\metals-loop.bat-12-set EXIT_CODE=%ERRORLEVEL%
--
scripts/win\pf-agent.bat-7-
scripts/win\pf-agent.bat-8-REM Clear Claude Code session markers — prevents "nested session" error when launched from
scripts/win\pf-agent.bat-9-REM a process tree that already has Claude Code running (e.g. Task Scheduler inheriting env)
scripts/win\pf-agent.bat:10:set CLAUDECODE=
scripts/win\pf-agent.bat:11:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\pf-agent.bat-12-
scripts/win\pf-agent.bat-13-REM Invoke Claude Code as the trading decision-maker
scripts/win\pf-agent.bat-14-REM Layer 1 already wrote fresh agent_summary.json — no need to re-collect
scripts/win\pf-agent.bat-15-echo Running trading agent...
scripts/win\pf-agent.bat:16:claude -p "You are the Layer 2 trading agent. FIRST read docs/TRADING_PLAYBOOK.md for trading rules. Then read data/layer2_context.md (your memory from previous invocations). Then read data/agent_summary_compact.json (signals, trigger reasons, timeframes), data/portfolio_state.json (Patient portfolio), and data/portfolio_state_bold.json (Bold portfolio). Follow the playbook to analyze, decide, and act for BOTH strategies independently. Compare your previous theses and prices with current data — were you right? Always write a journal entry and send a Telegram message." --allowedTools "Edit,Read,Bash,Write" --max-turns 40
--
scripts/win\oil-loop.bat-6-cd /d Q:\finance-analyzer
scripts/win\oil-loop.bat-7-
scripts/win\oil-loop.bat-8-:restart
scripts/win\oil-loop.bat:9:set CLAUDECODE=
scripts/win\oil-loop.bat:10:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\oil-loop.bat-11-echo [%date% %time%] Starting oil loop...
scripts/win\oil-loop.bat-12-.venv\Scripts\python.exe -u data\oil_loop.py --loop > data\oil_loop_out.txt 2>&1
scripts/win\oil-loop.bat-13-set EXIT_CODE=%ERRORLEVEL%
--
scripts/win\pf-outcome-check.bat-1-@echo off
scripts/win\pf-outcome-check.bat-2-REM PF-OutcomeCheck — Backfill price outcomes for signal accuracy tracking
scripts/win\pf-outcome-check.bat-3-cd /d Q:\finance-analyzer
scripts/win\pf-outcome-check.bat:4:set CLAUDECODE=
scripts/win\pf-outcome-check.bat:5:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\pf-outcome-check.bat-6-set PYTHONPATH=Q:\finance-analyzer
scripts/win\pf-outcome-check.bat-7-
scripts/win\pf-outcome-check.bat-8-echo [%date% %time%] Starting outcome backfill... >> data\outcome_check_out.txt 2>&1
--
scripts/win\pf-loop.bat-7-
scripts/win\pf-loop.bat-8-:restart
scripts/win\pf-loop.bat-9-REM Clear Claude Code session markers so Layer 2 agent can launch
scripts/win\pf-loop.bat:10:set CLAUDECODE=
scripts/win\pf-loop.bat:11:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\pf-loop.bat-12-set PYTHONPATH=Q:\finance-analyzer
scripts/win\pf-loop.bat-13-echo [%date% %time%] Starting loop...
scripts/win\pf-loop.bat-14-START /B /WAIT .venv\Scripts\python.exe -u portfolio\main.py --loop >> data\loop_out.txt 2>&1
--
scripts/win\rc-server-3.bat-1-@echo off
scripts/win\rc-server-3.bat-2-REM Claude Code Remote Control — Server 3 (Research)
scripts/win\rc-server-3.bat-3-cd /d Q:\finance-analyzer
scripts/win\rc-server-3.bat:4:set CLAUDECODE=
scripts/win\rc-server-3.bat:5:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\rc-server-3.bat-6-
scripts/win\rc-server-3.bat-7-:restart
scripts/win\rc-server-3.bat-8-echo [%date% %time%] Starting RC server 3 (Research)... >> data\rc-server-3_out.txt 2>&1
scripts/win\rc-server-3.bat:9:claude remote-control --name "Research" --spawn worktree --capacity 4 >> data\rc-server-3_out.txt 2>&1
scripts/win\rc-server-3.bat-10-echo [%date% %time%] RC server 3 exited (code %ERRORLEVEL%). Restarting in 15s... >> data\rc-server-3_out.txt 2>&1
scripts/win\rc-server-3.bat-11-timeout /t 15 /nobreak >nul
scripts/win\rc-server-3.bat-12-goto restart
--
scripts/win\rc-server-2.bat-1-@echo off
scripts/win\rc-server-2.bat-2-REM Claude Code Remote Control — Server 2 (Development)
scripts/win\rc-server-2.bat-3-cd /d Q:\finance-analyzer
scripts/win\rc-server-2.bat:4:set CLAUDECODE=
scripts/win\rc-server-2.bat:5:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\rc-server-2.bat-6-
scripts/win\rc-server-2.bat-7-:restart
scripts/win\rc-server-2.bat-8-echo [%date% %time%] Starting RC server 2 (Development)... >> data\rc-server-2_out.txt 2>&1
scripts/win\rc-server-2.bat:9:claude remote-control --name "Development" --spawn worktree --capacity 4 >> data\rc-server-2_out.txt 2>&1
scripts/win\rc-server-2.bat-10-echo [%date% %time%] RC server 2 exited (code %ERRORLEVEL%). Restarting in 15s... >> data\rc-server-2_out.txt 2>&1
scripts/win\rc-server-2.bat-11-timeout /t 15 /nobreak >nul
scripts/win\rc-server-2.bat-12-goto restart
--
scripts/win\rc-keepalive.ps1-3-# Problem: Anthropic server-side TTL is ~20 minutes. Only real user/model activity
scripts/win\rc-keepalive.ps1-4-# resets it - transport keepalives do NOT count. After ~20 min idle, the server deregisters
scripts/win\rc-keepalive.ps1-5-# the session. The CLI keeps polling (logs stay fresh), but the session becomes invisible
scripts/win\rc-keepalive.ps1:6:# in the claude.ai/code picker. Known bug: #28571, #29313, #34255, #37605, #38049.
scripts/win\rc-keepalive.ps1-7-#
scripts/win\rc-keepalive.ps1-8-# Solution: Every 5 min, check each RC server. If idle longer than its threshold
scripts/win\rc-keepalive.ps1-9-# (staggered: 13/15/17 min per server), kill it so the bat loop restarts fresh.
--
scripts/win\rc-keepalive.ps1-62-    Log "Wake-from-sleep mode - all idle servers will be recycled immediately."
scripts/win\rc-keepalive.ps1-63-}
scripts/win\rc-keepalive.ps1-64-
scripts/win\rc-keepalive.ps1:65:# Get all claude.exe processes
scripts/win\rc-keepalive.ps1:66:$allClaude = Get-CimInstance Win32_Process -Filter "Name='claude.exe'" -ErrorAction SilentlyContinue
scripts/win\rc-keepalive.ps1:67:$rcProcs = $allClaude | Where-Object { $_.CommandLine -match 'remote-control' }
scripts/win\rc-keepalive.ps1-68-$childProcs = $allClaude | Where-Object { $_.CommandLine -match 'session-id' }
scripts/win\rc-keepalive.ps1-69-
scripts/win\rc-keepalive.ps1-70-if (-not $rcProcs) {
--
scripts/win\rc-watchdog.ps1-46-}
scripts/win\rc-watchdog.ps1-47-
scripts/win\rc-watchdog.ps1-48-# --- Gather state ---
scripts/win\rc-watchdog.ps1:49:$rcProcs = Get-CimInstance Win32_Process -Filter "Name='claude.exe'" -ErrorAction SilentlyContinue |
scripts/win\rc-watchdog.ps1:50:    Where-Object { $_.CommandLine -match 'remote-control' }
scripts/win\rc-watchdog.ps1-51-
scripts/win\rc-watchdog.ps1-52-$tcpConns = Get-NetTCPConnection -State Established -RemotePort 443 -ErrorAction SilentlyContinue
scripts/win\rc-watchdog.ps1-53-
--
scripts/win\rc-watchdog.ps1-73-
scripts/win\rc-watchdog.ps1-74-    if (-not $proc) {
scripts/win\rc-watchdog.ps1-75-        if ($bat) {
scripts/win\rc-watchdog.ps1:76:            Log "$($srv.Name): claude.exe gone, bat loop alive - will auto-restart"
scripts/win\rc-watchdog.ps1-77-            $healthy++
scripts/win\rc-watchdog.ps1-78-        } else {
scripts/win\rc-watchdog.ps1-79-            Log "$($srv.Name): not running. Launching $($srv.BatFile)..."
--
scripts/win\rc-server-ensure.ps1-5-#   1. Log heartbeat: each server writes "Reconnected" to its output file every 2-3 min.
scripts/win\rc-server-ensure.ps1-6-#      If the log file hasn't been modified in 10+ min, the server is stale — even if the
scripts/win\rc-server-ensure.ps1-7-#      process is alive and has a TCP socket open. This catches the case where a session
scripts/win\rc-server-ensure.ps1:8:#      looks healthy by TCP but is no longer registered with Anthropic's remote-control API.
scripts/win\rc-server-ensure.ps1-9-#   2. TCP fallback: if the log file doesn't exist yet (first launch), fall back to checking
scripts/win\rc-server-ensure.ps1-10-#      for an ESTABLISHED TCP connection to port 443.
scripts/win\rc-server-ensure.ps1-11-#
--
scripts/win\rc-server-ensure.ps1-60-    Start-Sleep -Seconds 30
scripts/win\rc-server-ensure.ps1-61-}
scripts/win\rc-server-ensure.ps1-62-
scripts/win\rc-server-ensure.ps1:63:# Get all claude remote-control processes once
scripts/win\rc-server-ensure.ps1:64:$rcProcs = Get-CimInstance Win32_Process -Filter "Name='claude.exe'" -ErrorAction SilentlyContinue |
scripts/win\rc-server-ensure.ps1:65:    Where-Object { $_.CommandLine -match 'remote-control' }
scripts/win\rc-server-ensure.ps1-66-
scripts/win\rc-server-ensure.ps1-67-# Get all ESTABLISHED TCP connections to port 443 (Anthropic API) — used as fallback
scripts/win\rc-server-ensure.ps1-68-$tcpConns = Get-NetTCPConnection -State Established -RemotePort 443 -ErrorAction SilentlyContinue
--
scripts/win\rc-server-ensure.ps1-90-    return @{ Alive = $false; Method = "tcp"; Detail = "no connection and no log file" }
scripts/win\rc-server-ensure.ps1-91-}
scripts/win\rc-server-ensure.ps1-92-
scripts/win\rc-server-ensure.ps1:93:# Also find bat-loop cmd.exe processes (parents of the claude.exe servers)
scripts/win\rc-server-ensure.ps1-94-$batProcs = Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" -ErrorAction SilentlyContinue |
scripts/win\rc-server-ensure.ps1-95-    Where-Object { $_.CommandLine -match 'rc-server' }
scripts/win\rc-server-ensure.ps1-96-
--
scripts/win\rc-server-ensure.ps1-125-        }
scripts/win\rc-server-ensure.ps1-126-    } else {
scripts/win\rc-server-ensure.ps1-127-        if ($batRunning) {
scripts/win\rc-server-ensure.ps1:128:            # claude.exe died but bat loop is alive; it will restart on its own
scripts/win\rc-server-ensure.ps1:129:            Log "$($srv.Name) claude.exe gone but bat loop alive (PID $($batRunning.ProcessId)). Will auto-restart."
scripts/win\rc-server-ensure.ps1-130-            $skipped++
scripts/win\rc-server-ensure.ps1-131-        } else {
scripts/win\rc-server-ensure.ps1-132-            Log "$($srv.Name) not running. Launching $($srv.Bat)..."
--
scripts/win\rc-server.bat-5-cd /d Q:\finance-analyzer
scripts/win\rc-server.bat-6-
scripts/win\rc-server.bat-7-REM Clear Claude Code session markers (prevents nested-session errors)
scripts/win\rc-server.bat:8:set CLAUDECODE=
scripts/win\rc-server.bat:9:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\rc-server.bat-10-
scripts/win\rc-server.bat-11-:restart
scripts/win\rc-server.bat-12-echo [%date% %time%] Starting RC server 1 (Trading)... >> data\rc-server_out.txt 2>&1
scripts/win\rc-server.bat:13:claude remote-control --name "Trading" --spawn worktree --capacity 4 >> data\rc-server_out.txt 2>&1
scripts/win\rc-server.bat-14-echo [%date% %time%] RC server 1 exited (code %ERRORLEVEL%). Restarting in 15s... >> data\rc-server_out.txt 2>&1
scripts/win\rc-server.bat-15-timeout /t 15 /nobreak >nul
scripts/win\rc-server.bat-16-goto restart
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 765ms:
scripts/win\crypto-loop.bat-6-cd /d Q:\finance-analyzer
scripts/win\crypto-loop.bat-7-
scripts/win\crypto-loop.bat-8-:restart
scripts/win\crypto-loop.bat-9-REM Clear Claude Code session markers so any subagent invocation can launch.
scripts/win\crypto-loop.bat:10:set CLAUDECODE=
scripts/win\crypto-loop.bat:11:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\crypto-loop.bat-12-echo [%date% %time%] Starting crypto loop...
scripts/win\crypto-loop.bat-13-.venv\Scripts\python.exe -u data\crypto_loop.py --loop > data\crypto_loop_out.txt 2>&1
scripts/win\crypto-loop.bat-14-set EXIT_CODE=%ERRORLEVEL%
scripts/win\crypto-loop.bat-15-echo [%date% %time%] Crypto loop exited (code %EXIT_CODE%).
--
scripts/win\adversarial-review.bat-2-REM PF-AdversarialReview — Daily dual adversarial review (Codex + Claude)
scripts/win\adversarial-review.bat-3-REM Runs claude code CLI with the full review prompt.
scripts/win\adversarial-review.bat-4-REM Output: data\adversarial_review_out.txt
scripts/win\adversarial-review.bat-5-cd /d Q:\finance-analyzer
scripts/win\adversarial-review.bat:6:set CLAUDECODE=
scripts/win\adversarial-review.bat:7:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\adversarial-review.bat-8-
scripts/win\adversarial-review.bat-9-echo [%date% %time%] Starting adversarial review... >> data\adversarial_review_out.txt 2>&1
scripts/win\adversarial-review.bat-10-
scripts/win\adversarial-review.bat-11-claude -p "Follow /fgl protocol. Run a full dual adversarial review of the finance-analyzer codebase: partition into 8 subsystems (signals-core, orchestration, portfolio-risk, metals-core, avanza-api, signals-modules, data-external, infrastructure), create worktree with empty-baseline branches, run /codex:adversarial-review in background for each subsystem, write your own independent adversarial review, collect codex results, cross-critique in both directions, write synthesis doc. Commit all docs to main, push via Windows git, clean up worktrees. Do NOT ask for approval — follow /fgl rules. Spend your entire context on this." --allowedTools "Edit,Read,Bash,Write,Glob,Grep" --max-turns 80 >> data\adversarial_review_out.txt 2>&1
--
scripts/win\install-rc-server-task.ps1-8-# ---------- Task 1: Logon trigger (no delay) ----------
scripts/win\install-rc-server-task.ps1-9-$logonTask = "PF-RemoteControl"
scripts/win\install-rc-server-task.ps1-10-Unregister-ScheduledTask -TaskName $logonTask -Confirm:$false -ErrorAction SilentlyContinue
scripts/win\install-rc-server-task.ps1-11-
scripts/win\install-rc-server-task.ps1:12:$logonAction = New-ScheduledTaskAction `
scripts/win\install-rc-server-task.ps1-13-    -Execute "powershell.exe" `
scripts/win\install-rc-server-task.ps1:14:    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
scripts/win\install-rc-server-task.ps1-15-
scripts/win\install-rc-server-task.ps1-16-$logonTrigger = New-ScheduledTaskTrigger -AtLogOn
scripts/win\install-rc-server-task.ps1-17-
scripts/win\install-rc-server-task.ps1-18-$settings = New-ScheduledTaskSettingsSet `
--
scripts/win\install-rc-server-task.ps1-34-# ---------- Task 2: Wake-from-sleep trigger (with -WakeDelay) ----------
scripts/win\install-rc-server-task.ps1-35-$wakeTask = "PF-RemoteControl-Wake"
scripts/win\install-rc-server-task.ps1-36-Unregister-ScheduledTask -TaskName $wakeTask -Confirm:$false -ErrorAction SilentlyContinue
scripts/win\install-rc-server-task.ps1-37-
scripts/win\install-rc-server-task.ps1:38:$wakeAction = New-ScheduledTaskAction `
scripts/win\install-rc-server-task.ps1-39-    -Execute "powershell.exe" `
scripts/win\install-rc-server-task.ps1:40:    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" -WakeDelay"
scripts/win\install-rc-server-task.ps1-41-
scripts/win\install-rc-server-task.ps1-42-# Register with a placeholder trigger first, then replace via XML for event trigger
scripts/win\install-rc-server-task.ps1-43-$placeholderTrigger = New-ScheduledTaskTrigger -AtLogOn
scripts/win\install-rc-server-task.ps1-44-Register-ScheduledTask `
--
scripts/win\install-log-rotate-task.ps1-1-# Install the scheduled task that runs portfolio/log_rotation.py hourly.
scripts/win\install-log-rotate-task.ps1-2-# Run as:
scripts/win\install-log-rotate-task.ps1:3:#   powershell -ExecutionPolicy Bypass -File scripts\win\install-log-rotate-task.ps1
scripts/win\install-log-rotate-task.ps1-4-#
scripts/win\install-log-rotate-task.ps1-5-# Why hourly: loop_out.txt grows ~600KB/h at full verbosity. With a 5MB
scripts/win\install-log-rotate-task.ps1-6-# rotate threshold that's a ~8h cycle, so an hourly check keeps total
scripts/win\install-log-rotate-task.ps1-7-# (live + .1 .. .5.gz) under ~30MB. JSONL files in the policy mostly
--
scripts/win\install-log-rotate-task.ps1-19-    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
scripts/win\install-log-rotate-task.ps1-20-    Write-Host "Removed existing $TaskName"
scripts/win\install-log-rotate-task.ps1-21-}
scripts/win\install-log-rotate-task.ps1-22-
scripts/win\install-log-rotate-task.ps1:23:$action = New-ScheduledTaskAction -Execute $pythonExe `
scripts/win\install-log-rotate-task.ps1-24-    -Argument "-m portfolio.log_rotation" `
scripts/win\install-log-rotate-task.ps1-25-    -WorkingDirectory $workDir
scripts/win\install-log-rotate-task.ps1-26-
scripts/win\install-log-rotate-task.ps1-27-# Hourly trigger starting in 5 minutes (so first run isn't immediately
--
scripts/win\install-mstr-loop-task.ps1-1-# Install the canonical scheduled task for the MSTR loop.
scripts/win\install-mstr-loop-task.ps1-2-# Run as:
scripts/win\install-mstr-loop-task.ps1:3:#   powershell -ExecutionPolicy Bypass -File scripts\win\install-mstr-loop-task.ps1
scripts/win\install-mstr-loop-task.ps1-4-#
scripts/win\install-mstr-loop-task.ps1-5-# Defaults to PHASE=shadow via MSTR_LOOP_PHASE env var (per
scripts/win\install-mstr-loop-task.ps1-6-# docs/MSTR_LOOP_NOTES.md). Phase A (live) requires 90 days of shadow data
scripts/win\install-mstr-loop-task.ps1-7-# and explicit user approval — flip MSTR_LOOP_PHASE=live in the task's
--
scripts/win\install-mstr-loop-task.ps1-15-    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
scripts/win\install-mstr-loop-task.ps1-16-    Write-Host "Removed existing $TaskName"
scripts/win\install-mstr-loop-task.ps1-17-}
scripts/win\install-mstr-loop-task.ps1-18-
scripts/win\install-mstr-loop-task.ps1:19:# We use cmd /c with `set` to inject the phase env var before launching the
scripts/win\install-mstr-loop-task.ps1-20-# wrapper. Going via cmd avoids the PowerShell-vs-batch quoting hell.
scripts/win\install-mstr-loop-task.ps1:21:$action = New-ScheduledTaskAction -Execute "cmd.exe" `
scripts/win\install-mstr-loop-task.ps1:22:    -Argument "/c `"set `"MSTR_LOOP_PHASE=shadow`"&&`"$scriptDir\mstr-loop.bat`"`"" `
scripts/win\install-mstr-loop-task.ps1-23-    -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-mstr-loop-task.ps1-24-
scripts/win\install-mstr-loop-task.ps1-25-$trigger1 = New-ScheduledTaskTrigger -AtLogOn
scripts/win\install-mstr-loop-task.ps1-26-# MSTR is US-listed — only relevant during US session (15:30-22:00 CET).
--
scripts/win\install-loop-health-report-task.ps1-1-# One-time scheduled task: paper-mode health check 2 weeks after the
scripts/win\install-loop-health-report-task.ps1-2-# 2026-05-01 midfinance merge.
scripts/win\install-loop-health-report-task.ps1-3-#
scripts/win\install-loop-health-report-task.ps1-4-# Run as:
scripts/win\install-loop-health-report-task.ps1:5:#   powershell -ExecutionPolicy Bypass -File scripts\win\install-loop-health-report-task.ps1
scripts/win\install-loop-health-report-task.ps1-6-#
scripts/win\install-loop-health-report-task.ps1-7-# Fires once on 2026-05-15 at 18:00 local time (post-EU close, before
scripts/win\install-loop-health-report-task.ps1-8-# US close). The task auto-disables itself after firing. To re-arm for
scripts/win\install-loop-health-report-task.ps1-9-# a later date, edit $RunOnce below and re-run this script.
--
scripts/win\install-loop-health-report-task.ps1-16-    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
scripts/win\install-loop-health-report-task.ps1-17-    Write-Host "Removed existing $TaskName"
scripts/win\install-loop-health-report-task.ps1-18-}
scripts/win\install-loop-health-report-task.ps1-19-
scripts/win\install-loop-health-report-task.ps1:20:$action = New-ScheduledTaskAction -Execute "Q:\finance-analyzer\.venv\Scripts\python.exe" `
scripts/win\install-loop-health-report-task.ps1-21-    -Argument "Q:\finance-analyzer\scripts\loop_health_report.py" `
scripts/win\install-loop-health-report-task.ps1-22-    -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-loop-health-report-task.ps1-23-
scripts/win\install-loop-health-report-task.ps1-24-$trigger = New-ScheduledTaskTrigger -Once -At $RunOnce
--
scripts/win\install-rc-keepalive-task.ps1-13-
scripts/win\install-rc-keepalive-task.ps1-14-# Remove existing task if present
scripts/win\install-rc-keepalive-task.ps1-15-Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
scripts/win\install-rc-keepalive-task.ps1-16-
scripts/win\install-rc-keepalive-task.ps1:17:$actionPeriodic = New-ScheduledTaskAction `
scripts/win\install-rc-keepalive-task.ps1-18-    -Execute "powershell.exe" `
scripts/win\install-rc-keepalive-task.ps1:19:    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
scripts/win\install-rc-keepalive-task.ps1-20-
scripts/win\install-rc-keepalive-task.ps1:21:$actionWake = New-ScheduledTaskAction `
scripts/win\install-rc-keepalive-task.ps1-22-    -Execute "powershell.exe" `
scripts/win\install-rc-keepalive-task.ps1:23:    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" -Wake"
scripts/win\install-rc-keepalive-task.ps1-24-
scripts/win\install-rc-keepalive-task.ps1-25-# Trigger 1: every 5 minutes, starting now, repeating for ~25 years (max safe duration)
scripts/win\install-rc-keepalive-task.ps1-26-$triggerPeriodic = New-ScheduledTaskTrigger -Once -At (Get-Date) `
scripts/win\install-rc-keepalive-task.ps1-27-    -RepetitionInterval (New-TimeSpan -Minutes 5) `
--
scripts/win\install-local-llm-report-task.ps1-4-    [int]$Days = 30,
scripts/win\install-local-llm-report-task.ps1-5-    [switch]$Remove
scripts/win\install-local-llm-report-task.ps1-6-)
scripts/win\install-local-llm-report-task.ps1-7-
scripts/win\install-local-llm-report-task.ps1:8:$ErrorActionPreference = "Stop"
scripts/win\install-local-llm-report-task.ps1-9-
scripts/win\install-local-llm-report-task.ps1-10-if ($Time -notmatch '^\d{2}:\d{2}$') {
scripts/win\install-local-llm-report-task.ps1-11-    throw "Time must be in HH:mm format."
scripts/win\install-local-llm-report-task.ps1-12-}
--
scripts/win\install-local-llm-report-task.ps1-28-}
scripts/win\install-local-llm-report-task.ps1-29-
scripts/win\install-local-llm-report-task.ps1-30-$taskCommand = '"' + $runner + '" ' + $Days
scripts/win\install-local-llm-report-task.ps1-31-
scripts/win\install-local-llm-report-task.ps1:32:& schtasks /Create /TN $TaskName /SC DAILY /ST $Time /TR $taskCommand /F
scripts/win\install-local-llm-report-task.ps1-33-if ($LASTEXITCODE -ne 0) {
scripts/win\install-local-llm-report-task.ps1-34-    exit $LASTEXITCODE
scripts/win\install-local-llm-report-task.ps1-35-}
scripts/win\install-local-llm-report-task.ps1-36-
--
scripts/win\meta-learner-retrain.bat-1-@echo off
scripts/win\meta-learner-retrain.bat-2-REM PF-MetaLearnerRetrain — Daily LightGBM meta-learner retraining (low priority)
scripts/win\meta-learner-retrain.bat-3-cd /d Q:\finance-analyzer
scripts/win\meta-learner-retrain.bat:4:set CLAUDECODE=
scripts/win\meta-learner-retrain.bat:5:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\meta-learner-retrain.bat-6-set PYTHONPATH=Q:\finance-analyzer
scripts/win\meta-learner-retrain.bat-7-
scripts/win\meta-learner-retrain.bat-8-echo [%date% %time%] Starting meta-learner retraining... >> data\meta_learner_retrain_out.txt 2>&1
scripts/win\meta-learner-retrain.bat-9-start /LOW /B .venv\Scripts\python.exe -u portfolio/meta_learner.py >> data\meta_learner_retrain_out.txt 2>&1
--
scripts/win\install-crypto-loop-task.ps1-1-# Install the canonical scheduled task for the crypto (BTC+ETH) loop.
scripts/win\install-crypto-loop-task.ps1-2-# Run as:
scripts/win\install-crypto-loop-task.ps1:3:#   powershell -ExecutionPolicy Bypass -File scripts\win\install-crypto-loop-task.ps1
scripts/win\install-crypto-loop-task.ps1-4-#
scripts/win\install-crypto-loop-task.ps1-5-# Parity with PF-MetalsLoop:
scripts/win\install-crypto-loop-task.ps1-6-#   - AtLogOn + weekly Mon-Fri 07:00 triggers
scripts/win\install-crypto-loop-task.ps1-7-#   - 3-day execution time limit (the wrapper auto-restarts on crash)
--
scripts/win\install-crypto-loop-task.ps1-21-    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
scripts/win\install-crypto-loop-task.ps1-22-    Write-Host "Removed existing $TaskName"
scripts/win\install-crypto-loop-task.ps1-23-}
scripts/win\install-crypto-loop-task.ps1-24-
scripts/win\install-crypto-loop-task.ps1:25:$action = New-ScheduledTaskAction -Execute "cmd.exe" `
scripts/win\install-crypto-loop-task.ps1:26:    -Argument "/c `"$scriptDir\crypto-loop.bat`"" `
scripts/win\install-crypto-loop-task.ps1-27-    -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-crypto-loop-task.ps1-28-
scripts/win\install-crypto-loop-task.ps1-29-$trigger1 = New-ScheduledTaskTrigger -AtLogOn
scripts/win\install-crypto-loop-task.ps1-30-$trigger2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday,Saturday,Sunday -At "07:00"
--
scripts/win\install-golddigger-task.ps1-2-    [string]$TaskName = "PF-GoldDigger",
scripts/win\install-golddigger-task.ps1-3-    [switch]$Remove
scripts/win\install-golddigger-task.ps1-4-)
scripts/win\install-golddigger-task.ps1-5-
scripts/win\install-golddigger-task.ps1:6:$ErrorActionPreference = "Stop"
scripts/win\install-golddigger-task.ps1-7-
scripts/win\install-golddigger-task.ps1-8-$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
scripts/win\install-golddigger-task.ps1-9-$runner = Join-Path $repoRoot "scripts\win\golddigger.bat"
scripts/win\install-golddigger-task.ps1-10-
--
scripts/win\install-golddigger-task.ps1-18-}
scripts/win\install-golddigger-task.ps1-19-
scripts/win\install-golddigger-task.ps1-20-$taskCommand = '"' + $runner + '" --live'
scripts/win\install-golddigger-task.ps1-21-
scripts/win\install-golddigger-task.ps1:22:& schtasks /Create /TN $TaskName /SC ONLOGON /TR $taskCommand /F
scripts/win\install-golddigger-task.ps1-23-if ($LASTEXITCODE -ne 0) {
scripts/win\install-golddigger-task.ps1-24-    exit $LASTEXITCODE
scripts/win\install-golddigger-task.ps1-25-}
scripts/win\install-golddigger-task.ps1-26-
--
scripts/win\install-health-check-tasks.ps1-5-$script = "Q:\finance-analyzer\scripts\health_check.py"
scripts/win\install-health-check-tasks.ps1-6-$workDir = "Q:\finance-analyzer"
scripts/win\install-health-check-tasks.ps1-7-
scripts/win\install-health-check-tasks.ps1-8-# Tier 1: Full check at 11:00 CET (09:00 UTC summer / 10:00 UTC winter)
scripts/win\install-health-check-tasks.ps1:9:$action1 = New-ScheduledTaskAction -Execute $pythonExe -Argument "-u $script --tier full" -WorkingDirectory $workDir
scripts/win\install-health-check-tasks.ps1-10-$trigger1 = New-ScheduledTaskTrigger -Daily -At "11:00"
scripts/win\install-health-check-tasks.ps1-11-$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
scripts/win\install-health-check-tasks.ps1-12-Register-ScheduledTask -TaskName "PF-HealthCheck-Full" -Action $action1 -Trigger $trigger1 -Settings $settings -Description "System health contract - full check (11:00 CET)" -Force
scripts/win\install-health-check-tasks.ps1-13-
scripts/win\install-health-check-tasks.ps1-14-# Tier 2: Pre-US check at 15:25 CET (13:25 UTC summer / 14:25 UTC winter)
scripts/win\install-health-check-tasks.ps1:15:$action2 = New-ScheduledTaskAction -Execute $pythonExe -Argument "-u $script --tier pre-us" -WorkingDirectory $workDir
scripts/win\install-health-check-tasks.ps1-16-$trigger2 = New-ScheduledTaskTrigger -Daily -At "15:25"
scripts/win\install-health-check-tasks.ps1-17-Register-ScheduledTask -TaskName "PF-HealthCheck-PreUS" -Action $action2 -Trigger $trigger2 -Settings $settings -Description "System health contract - pre-US-open check (15:25 CET)" -Force
scripts/win\install-health-check-tasks.ps1-18-
scripts/win\install-health-check-tasks.ps1-19-# Tier 3: Post-US check at 22:05 CET (20:05 UTC summer / 21:05 UTC winter)
scripts/win\install-health-check-tasks.ps1:20:$action3 = New-ScheduledTaskAction -Execute $pythonExe -Argument "-u $script --tier post-us" -WorkingDirectory $workDir
scripts/win\install-health-check-tasks.ps1-21-$trigger3 = New-ScheduledTaskTrigger -Daily -At "22:05"
scripts/win\install-health-check-tasks.ps1-22-Register-ScheduledTask -TaskName "PF-HealthCheck-PostUS" -Action $action3 -Trigger $trigger3 -Settings $settings -Description "System health contract - post-US-close check (22:05 CET)" -Force
scripts/win\install-health-check-tasks.ps1-23-
scripts/win\install-health-check-tasks.ps1-24-Write-Host "Installed 3 health check tasks:"
--
scripts/win\install-metals-loop-task.ps1-1-# Install the canonical scheduled task for the brokered metals loop.
scripts/win\install-metals-loop-task.ps1-2-# Run as:
scripts/win\install-metals-loop-task.ps1:3:#   powershell -ExecutionPolicy Bypass -File scripts\win\install-metals-loop-task.ps1
scripts/win\install-metals-loop-task.ps1-4-
scripts/win\install-metals-loop-task.ps1-5-$TaskName = "PF-MetalsLoop"
scripts/win\install-metals-loop-task.ps1-6-$scriptDir = "Q:\finance-analyzer\scripts\win"
scripts/win\install-metals-loop-task.ps1-7-
--
scripts/win\install-metals-loop-task.ps1-10-    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
scripts/win\install-metals-loop-task.ps1-11-    Write-Host "Removed existing $TaskName"
scripts/win\install-metals-loop-task.ps1-12-}
scripts/win\install-metals-loop-task.ps1-13-
scripts/win\install-metals-loop-task.ps1:14:$action = New-ScheduledTaskAction -Execute "cmd.exe" `
scripts/win\install-metals-loop-task.ps1:15:    -Argument "/c `"$scriptDir\metals-loop.bat`"" `
scripts/win\install-metals-loop-task.ps1-16-    -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-metals-loop-task.ps1-17-
scripts/win\install-metals-loop-task.ps1-18-$trigger1 = New-ScheduledTaskTrigger -AtLogOn
scripts/win\install-metals-loop-task.ps1-19-$trigger2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "07:00"
--
scripts/win\install-signal-research-task.ps1-1-# Install PF-SignalResearch scheduled task
scripts/win\install-signal-research-task.ps1:2:# Run as: powershell -ExecutionPolicy Bypass -File scripts\win\install-signal-research-task.ps1
scripts/win\install-signal-research-task.ps1-3-#
scripts/win\install-signal-research-task.ps1-4-# Schedule:
scripts/win\install-signal-research-task.ps1-5-#   Daily at 18:30 CET (after EU market close, before after-hours research at 22:30)
scripts/win\install-signal-research-task.ps1-6-#   Runs Claude Code CLI with signal research prompt
--
scripts/win\install-signal-research-task.ps1-15-    Write-Host "Removed existing task: $taskName"
scripts/win\install-signal-research-task.ps1-16-}
scripts/win\install-signal-research-task.ps1-17-
scripts/win\install-signal-research-task.ps1-18-# Create the action
scripts/win\install-signal-research-task.ps1:19:$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$scriptPath`"" -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-signal-research-task.ps1-20-
scripts/win\install-signal-research-task.ps1-21-# Trigger: daily at 18:30 (after EU close, before after-hours at 22:30)
scripts/win\install-signal-research-task.ps1-22-$trigger = New-ScheduledTaskTrigger -Daily -At "18:30"
scripts/win\install-signal-research-task.ps1-23-
--
scripts/win\install-oil-loop-task.ps1-1-# Install the canonical scheduled task for the oil (WTI) loop.
scripts/win\install-oil-loop-task.ps1-2-# Run as:
scripts/win\install-oil-loop-task.ps1:3:#   powershell -ExecutionPolicy Bypass -File scripts\win\install-oil-loop-task.ps1
scripts/win\install-oil-loop-task.ps1-4-#
scripts/win\install-oil-loop-task.ps1-5-# Parity with PF-CryptoLoop / PF-MetalsLoop. Oil futures trade nearly 24/7
scripts/win\install-oil-loop-task.ps1-6-# on CME (Sun 23:00 CET to Fri 22:00 CET), so the weekly trigger covers
scripts/win\install-oil-loop-task.ps1-7-# Mon-Fri (Saturday is the only fully closed day).
--
scripts/win\install-oil-loop-task.ps1-14-    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
scripts/win\install-oil-loop-task.ps1-15-    Write-Host "Removed existing $TaskName"
scripts/win\install-oil-loop-task.ps1-16-}
scripts/win\install-oil-loop-task.ps1-17-
scripts/win\install-oil-loop-task.ps1:18:$action = New-ScheduledTaskAction -Execute "cmd.exe" `
scripts/win\install-oil-loop-task.ps1:19:    -Argument "/c `"$scriptDir\oil-loop.bat`"" `
scripts/win\install-oil-loop-task.ps1-20-    -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-oil-loop-task.ps1-21-
scripts/win\install-oil-loop-task.ps1-22-$trigger1 = New-ScheduledTaskTrigger -AtLogOn
scripts/win\install-oil-loop-task.ps1-23-$trigger2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday,Monday,Tuesday,Wednesday,Thursday,Friday -At "07:00"
--
scripts/win\install-market-tasks.ps1-1-# Install scheduled tasks for Silver Monitor and GoldDigger
scripts/win\install-market-tasks.ps1:2:# Run as: powershell -ExecutionPolicy Bypass -File scripts\win\install-market-tasks.ps1
scripts/win\install-market-tasks.ps1-3-#
scripts/win\install-market-tasks.ps1-4-# Schedule:
scripts/win\install-market-tasks.ps1-5-#   Start: 07:00 CET daily (Mon-Fri) = EU market pre-open
scripts/win\install-market-tasks.ps1-6-#   Auto-stop: bat files exit after 22:00 CET
--
scripts/win\install-market-tasks.ps1-19-    Unregister-ScheduledTask -TaskName $taskName1 -Confirm:$false
scripts/win\install-market-tasks.ps1-20-    Write-Host "Removed existing $taskName1"
scripts/win\install-market-tasks.ps1-21-}
scripts/win\install-market-tasks.ps1-22-
scripts/win\install-market-tasks.ps1:23:$action1 = New-ScheduledTaskAction -Execute "cmd.exe" `
scripts/win\install-market-tasks.ps1:24:    -Argument "/c `"$scriptDir\silver-monitor.bat`"" `
scripts/win\install-market-tasks.ps1-25-    -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-market-tasks.ps1-26-
scripts/win\install-market-tasks.ps1-27-$trigger1 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "07:00"
scripts/win\install-market-tasks.ps1-28-$trigger1.Repetition = $null
--
scripts/win\install-market-tasks.ps1-49-    Unregister-ScheduledTask -TaskName $taskName2 -Confirm:$false
scripts/win\install-market-tasks.ps1-50-    Write-Host "Removed existing $taskName2"
scripts/win\install-market-tasks.ps1-51-}
scripts/win\install-market-tasks.ps1-52-
scripts/win\install-market-tasks.ps1:53:$action2 = New-ScheduledTaskAction -Execute "cmd.exe" `
scripts/win\install-market-tasks.ps1:54:    -Argument "/c `"$scriptDir\golddigger-loop.bat`"" `
scripts/win\install-market-tasks.ps1-55-    -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-market-tasks.ps1-56-
scripts/win\install-market-tasks.ps1-57-$trigger2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "07:00"
scripts/win\install-market-tasks.ps1-58-
--
scripts/win\install-loop-health-daily-task.ps1-1-# Install the DAILY loop-health summary scheduled task.
scripts/win\install-loop-health-daily-task.ps1-2-#
scripts/win\install-loop-health-daily-task.ps1-3-# Run as:
scripts/win\install-loop-health-daily-task.ps1:4:#   powershell -ExecutionPolicy Bypass -File scripts\win\install-loop-health-daily-task.ps1
scripts/win\install-loop-health-daily-task.ps1-5-#
scripts/win\install-loop-health-daily-task.ps1-6-# Fires every morning at 08:00 local time. Sends a Telegram summary with:
scripts/win\install-loop-health-daily-task.ps1-7-#   - heartbeat freshness for crypto + oil
scripts/win\install-loop-health-daily-task.ps1-8-#   - mstr_loop_poll.jsonl existence (mstr proxy)
--
scripts/win\install-loop-health-daily-task.ps1-27-    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
scripts/win\install-loop-health-daily-task.ps1-28-    Write-Host "Removed existing $TaskName"
scripts/win\install-loop-health-daily-task.ps1-29-}
scripts/win\install-loop-health-daily-task.ps1-30-
scripts/win\install-loop-health-daily-task.ps1:31:$action = New-ScheduledTaskAction -Execute "Q:\finance-analyzer\.venv\Scripts\python.exe" `
scripts/win\install-loop-health-daily-task.ps1-32-    -Argument "$scriptDir\loop_health_report.py" `
scripts/win\install-loop-health-daily-task.ps1-33-    -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-loop-health-daily-task.ps1-34-
scripts/win\install-loop-health-daily-task.ps1-35-# 08:00 local — well after EU pre-open prep (07:00) so the loops have
--
scripts/win\install-adversarial-review-task.ps1-1-# Install PF-AdversarialReview scheduled task
scripts/win\install-adversarial-review-task.ps1:2:# Run as: powershell -ExecutionPolicy Bypass -File scripts\win\install-adversarial-review-task.ps1
scripts/win\install-adversarial-review-task.ps1-3-#
scripts/win\install-adversarial-review-task.ps1-4-# Schedule:
scripts/win\install-adversarial-review-task.ps1-5-#   Daily at 17:20 CET/CEST (after market close, before after-hours research)
scripts/win\install-adversarial-review-task.ps1-6-#   Runs Claude Code CLI with /fgl protocol for dual adversarial review
--
scripts/win\install-adversarial-review-task.ps1-14-    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
scripts/win\install-adversarial-review-task.ps1-15-    Write-Host "Removed existing task: $taskName"
scripts/win\install-adversarial-review-task.ps1-16-}
scripts/win\install-adversarial-review-task.ps1-17-
scripts/win\install-adversarial-review-task.ps1:18:$action = New-ScheduledTaskAction `
scripts/win\install-adversarial-review-task.ps1-19-    -Execute $scriptPath `
scripts/win\install-adversarial-review-task.ps1-20-    -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-adversarial-review-task.ps1-21-
scripts/win\install-adversarial-review-task.ps1-22-$trigger = New-ScheduledTaskTrigger -Daily -At "17:20"
--
scripts/win\install-fix-agent-task.ps1-19-    -RepetitionInterval (New-TimeSpan -Minutes 10) `
scripts/win\install-fix-agent-task.ps1-20-    -RepetitionDuration ([TimeSpan]::MaxValue)
scripts/win\install-fix-agent-task.ps1-21-
scripts/win\install-fix-agent-task.ps1-22-# Action: python scripts/fix_agent_dispatcher.py
scripts/win\install-fix-agent-task.ps1:23:$action = New-ScheduledTaskAction `
scripts/win\install-fix-agent-task.ps1-24-    -Execute $pythonPath `
scripts/win\install-fix-agent-task.ps1-25-    -Argument "-u `"$scriptPath`"" `
scripts/win\install-fix-agent-task.ps1-26-    -WorkingDirectory $workingDir
scripts/win\install-fix-agent-task.ps1-27-
--
scripts/win\install-meta-learner-task.ps1-1-# Install PF-MetaLearnerRetrain scheduled task
scripts/win\install-meta-learner-task.ps1:2:# Run as: powershell -ExecutionPolicy Bypass -File scripts\win\install-meta-learner-task.ps1
scripts/win\install-meta-learner-task.ps1-3-#
scripts/win\install-meta-learner-task.ps1-4-# Schedule:
scripts/win\install-meta-learner-task.ps1-5-#   Daily at 19:00 CET (after market close, after PF-OutcomeCheck at 18:00)
scripts/win\install-meta-learner-task.ps1-6-#   Low priority (start /LOW + os.nice(19) + num_threads=1) to avoid disrupting trading loop
--
scripts/win\install-meta-learner-task.ps1-14-    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
scripts/win\install-meta-learner-task.ps1-15-    Write-Host "Removed existing task: $taskName"
scripts/win\install-meta-learner-task.ps1-16-}
scripts/win\install-meta-learner-task.ps1-17-
scripts/win\install-meta-learner-task.ps1:18:$action = New-ScheduledTaskAction `
scripts/win\install-meta-learner-task.ps1-19-    -Execute $scriptPath `
scripts/win\install-meta-learner-task.ps1-20-    -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-meta-learner-task.ps1-21-
scripts/win\install-meta-learner-task.ps1-22-$trigger = New-ScheduledTaskTrigger -Daily -At "19:00"
--
scripts/win\install-rc-watchdog-task.ps1-8-
scripts/win\install-rc-watchdog-task.ps1-9-# Remove existing task if present
scripts/win\install-rc-watchdog-task.ps1-10-Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
scripts/win\install-rc-watchdog-task.ps1-11-
scripts/win\install-rc-watchdog-task.ps1:12:$action = New-ScheduledTaskAction `
scripts/win\install-rc-watchdog-task.ps1-13-    -Execute "powershell.exe" `
scripts/win\install-rc-watchdog-task.ps1:14:    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
scripts/win\install-rc-watchdog-task.ps1-15-
scripts/win\install-rc-watchdog-task.ps1-16-# Trigger: every 30 minutes, starting now, repeating indefinitely
scripts/win\install-rc-watchdog-task.ps1-17-$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `
scripts/win\install-rc-watchdog-task.ps1-18-    -RepetitionInterval (New-TimeSpan -Minutes 30) `
--
scripts/win\pf-agent.bat-6-cd /d Q:\finance-analyzer
scripts/win\pf-agent.bat-7-
scripts/win\pf-agent.bat-8-REM Clear Claude Code session markers — prevents "nested session" error when launched from
scripts/win\pf-agent.bat-9-REM a process tree that already has Claude Code running (e.g. Task Scheduler inheriting env)
scripts/win\pf-agent.bat:10:set CLAUDECODE=
scripts/win\pf-agent.bat:11:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\pf-agent.bat-12-
scripts/win\pf-agent.bat-13-REM Invoke Claude Code as the trading decision-maker
scripts/win\pf-agent.bat-14-REM Layer 1 already wrote fresh agent_summary.json — no need to re-collect
scripts/win\pf-agent.bat-15-echo Running trading agent...
--
scripts/win\metals-loop.bat-4-cd /d Q:\finance-analyzer
scripts/win\metals-loop.bat-5-
scripts/win\metals-loop.bat-6-:restart
scripts/win\metals-loop.bat-7-REM Clear Claude Code session markers so Layer 2 agent can launch
scripts/win\metals-loop.bat:8:set CLAUDECODE=
scripts/win\metals-loop.bat:9:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\metals-loop.bat-10-echo [%date% %time%] Starting metals loop...
scripts/win\metals-loop.bat-11-.venv\Scripts\python.exe -u data\metals_loop.py > data\metals_loop_out.txt 2>&1
scripts/win\metals-loop.bat-12-set EXIT_CODE=%ERRORLEVEL%
scripts/win\metals-loop.bat-13-echo [%date% %time%] Metals loop exited (code %EXIT_CODE%).
--
scripts/win\install-research-task.ps1-12-    Write-Host "Removed existing task: $taskName"
scripts/win\install-research-task.ps1-13-}
scripts/win\install-research-task.ps1-14-
scripts/win\install-research-task.ps1-15-# Create the action
scripts/win\install-research-task.ps1:16:$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$scriptPath`"" -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-research-task.ps1-17-
scripts/win\install-research-task.ps1-18-# Trigger: daily at 22:30 (10:30 PM local time — after US close)
scripts/win\install-research-task.ps1-19-$trigger = New-ScheduledTaskTrigger -Daily -At "22:30"
scripts/win\install-research-task.ps1-20-
--
scripts/win\oil-loop.bat-5-REM fork-bombing into the live instance.
scripts/win\oil-loop.bat-6-cd /d Q:\finance-analyzer
scripts/win\oil-loop.bat-7-
scripts/win\oil-loop.bat-8-:restart
scripts/win\oil-loop.bat:9:set CLAUDECODE=
scripts/win\oil-loop.bat:10:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\oil-loop.bat-11-echo [%date% %time%] Starting oil loop...
scripts/win\oil-loop.bat-12-.venv\Scripts\python.exe -u data\oil_loop.py --loop > data\oil_loop_out.txt 2>&1
scripts/win\oil-loop.bat-13-set EXIT_CODE=%ERRORLEVEL%
scripts/win\oil-loop.bat-14-echo [%date% %time%] Oil loop exited (code %EXIT_CODE%).
--
scripts/win\install-loop-resume-task.ps1-10-# Trigger: Event ID 1 from Microsoft-Windows-Power-Troubleshooter = resume from sleep
scripts/win\install-loop-resume-task.ps1-11-$trigger = New-ScheduledTaskTrigger -AtLogOn  # placeholder, replaced by XML below
scripts/win\install-loop-resume-task.ps1-12-
scripts/win\install-loop-resume-task.ps1-13-# Action: run the ensure script
scripts/win\install-loop-resume-task.ps1:14:$action = New-ScheduledTaskAction `
scripts/win\install-loop-resume-task.ps1-15-    -Execute "powershell.exe" `
scripts/win\install-loop-resume-task.ps1:16:    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
scripts/win\install-loop-resume-task.ps1-17-
scripts/win\install-loop-resume-task.ps1-18-# Settings
scripts/win\install-loop-resume-task.ps1-19-$settings = New-ScheduledTaskSettingsSet `
scripts/win\install-loop-resume-task.ps1-20-    -AllowStartIfOnBatteries `
--
scripts/win\install-loop-health-watchdog-task.ps1-1-# Install the periodic loop-health watchdog scheduled task.
scripts/win\install-loop-health-watchdog-task.ps1-2-#
scripts/win\install-loop-health-watchdog-task.ps1-3-# Run as:
scripts/win\install-loop-health-watchdog-task.ps1:4:#   powershell -ExecutionPolicy Bypass -File scripts\win\install-loop-health-watchdog-task.ps1
scripts/win\install-loop-health-watchdog-task.ps1-5-#
scripts/win\install-loop-health-watchdog-task.ps1-6-# Fires every 30 minutes from logon onward. Sends a consolidated telegram
scripts/win\install-loop-health-watchdog-task.ps1-7-# alert when any loop heartbeat is stale or missing. Per-loop cooldown
scripts/win\install-loop-health-watchdog-task.ps1-8-# (4h default in scripts/loop_health_watchdog.py) prevents alert spam
--
scripts/win\install-loop-health-watchdog-task.ps1-16-    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
scripts/win\install-loop-health-watchdog-task.ps1-17-    Write-Host "Removed existing $TaskName"
scripts/win\install-loop-health-watchdog-task.ps1-18-}
scripts/win\install-loop-health-watchdog-task.ps1-19-
scripts/win\install-loop-health-watchdog-task.ps1:20:$action = New-ScheduledTaskAction -Execute "Q:\finance-analyzer\.venv\Scripts\python.exe" `
scripts/win\install-loop-health-watchdog-task.ps1-21-    -Argument "$scriptDir\loop_health_watchdog.py" `
scripts/win\install-loop-health-watchdog-task.ps1-22-    -WorkingDirectory "Q:\finance-analyzer"
scripts/win\install-loop-health-watchdog-task.ps1-23-
scripts/win\install-loop-health-watchdog-task.ps1-24-# Trigger every 30 minutes, indefinitely, starting at logon
--
scripts/win\pf-loop-ensure.ps1-24-    exit 0
scripts/win\pf-loop-ensure.ps1-25-}
scripts/win\pf-loop-ensure.ps1-26-
scripts/win\pf-loop-ensure.ps1-27-Write-Host "[pf-loop-ensure] No loop detected. Starting pf-loop.bat..."
scripts/win\pf-loop-ensure.ps1:28:Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$batPath`"" -WindowStyle Minimized
scripts/win\pf-loop-ensure.ps1-29-Write-Host "[pf-loop-ensure] Launched pf-loop.bat at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
--
scripts/win\pf-loop.bat-6-cd /d Q:\finance-analyzer
scripts/win\pf-loop.bat-7-
scripts/win\pf-loop.bat-8-:restart
scripts/win\pf-loop.bat-9-REM Clear Claude Code session markers so Layer 2 agent can launch
scripts/win\pf-loop.bat:10:set CLAUDECODE=
scripts/win\pf-loop.bat:11:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\pf-loop.bat-12-set PYTHONPATH=Q:\finance-analyzer
scripts/win\pf-loop.bat-13-echo [%date% %time%] Starting loop...
scripts/win\pf-loop.bat-14-START /B /WAIT .venv\Scripts\python.exe -u portfolio\main.py --loop >> data\loop_out.txt 2>&1
scripts/win\pf-loop.bat-15-set EXIT_CODE=%ERRORLEVEL%
--
scripts/win\pf-outcome-check.bat-1-@echo off
scripts/win\pf-outcome-check.bat-2-REM PF-OutcomeCheck — Backfill price outcomes for signal accuracy tracking
scripts/win\pf-outcome-check.bat-3-cd /d Q:\finance-analyzer
scripts/win\pf-outcome-check.bat:4:set CLAUDECODE=
scripts/win\pf-outcome-check.bat:5:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\pf-outcome-check.bat-6-set PYTHONPATH=Q:\finance-analyzer
scripts/win\pf-outcome-check.bat-7-
scripts/win\pf-outcome-check.bat-8-echo [%date% %time%] Starting outcome backfill... >> data\outcome_check_out.txt 2>&1
scripts/win\pf-outcome-check.bat-9-.venv\Scripts\python.exe -u portfolio\main.py --check-outcomes >> data\outcome_check_out.txt 2>&1
--
scripts/win\pf-restart.bat-1-@echo off
scripts/win\pf-restart.bat:2:REM Thin wrapper around pf-restart.ps1 so the loop can be restarted from cmd.exe
scripts/win\pf-restart.bat:3:REM or `cmd.exe /c scripts\win\pf-restart.bat` from WSL. See pf-restart.ps1 for
scripts/win\pf-restart.bat-4-REM the full rationale on why schtasks /end leaves orphan python.exe children.
scripts/win\pf-restart.bat-5-REM
scripts/win\pf-restart.bat-6-REM Usage:
scripts/win\pf-restart.bat-7-REM   pf-restart.bat            (default: loop / PF-DataLoop)
--
scripts/win\pf-restart.bat-10-
scripts/win\pf-restart.bat-11-set TARGET=%~1
scripts/win\pf-restart.bat-12-if "%TARGET%"=="" set TARGET=loop
scripts/win\pf-restart.bat-13-
scripts/win\pf-restart.bat:14:powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0pf-restart.ps1" -Target %TARGET%
scripts/win\pf-restart.bat-15-exit /b %ERRORLEVEL%
--
scripts/win\pf-restart.ps1-65-    } else {
scripts/win\pf-restart.ps1-66-        Write-Host "    no running python.exe matching '$($t.Match)'"
scripts/win\pf-restart.ps1-67-    }
scripts/win\pf-restart.ps1-68-
scripts/win\pf-restart.ps1:69:    # Stop the scheduled task wrapper (cmd.exe running pf-*.bat). Safe even
scripts/win\pf-restart.ps1-70-    # if it's already gone - schtasks returns 1 in that case which we ignore.
scripts/win\pf-restart.ps1-71-    & schtasks.exe /end /tn $t.Task 2>&1 | Out-Null
scripts/win\pf-restart.ps1-72-
scripts/win\pf-restart.ps1-73-    # Start fresh
--
scripts/win\rc-keepalive.ps1-44-}
scripts/win\rc-keepalive.ps1-45-
scripts/win\rc-keepalive.ps1-46-function Send-Telegram($msg) {
scripts/win\rc-keepalive.ps1-47-    try {
scripts/win\rc-keepalive.ps1:48:        $cfg = Get-Content $configFile -Raw | ConvertFrom-Json
scripts/win\rc-keepalive.ps1-49-        $token  = $cfg.telegram.token
scripts/win\rc-keepalive.ps1-50-        $chatId = $cfg.telegram.chat_id
scripts/win\rc-keepalive.ps1-51-        if (-not $token -or -not $chatId) { return }
scripts/win\rc-keepalive.ps1-52-        $body = @{ chat_id = $chatId; text = $msg; parse_mode = "Markdown" } | ConvertTo-Json
scripts/win\rc-keepalive.ps1:53:        Invoke-RestMethod -Uri "https://api.telegram.org/bot$token/sendMessage" `
scripts/win\rc-keepalive.ps1-54-            -Method Post -ContentType "application/json" -Body $body `
scripts/win\rc-keepalive.ps1-55-            -TimeoutSec 15 | Out-Null
scripts/win\rc-keepalive.ps1-56-    } catch {
scripts/win\rc-keepalive.ps1-57-        Log "Telegram send failed: $_"
--
scripts/win\rc-server-2.bat-1-@echo off
scripts/win\rc-server-2.bat-2-REM Claude Code Remote Control — Server 2 (Development)
scripts/win\rc-server-2.bat-3-cd /d Q:\finance-analyzer
scripts/win\rc-server-2.bat:4:set CLAUDECODE=
scripts/win\rc-server-2.bat:5:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\rc-server-2.bat-6-
scripts/win\rc-server-2.bat-7-:restart
scripts/win\rc-server-2.bat-8-echo [%date% %time%] Starting RC server 2 (Development)... >> data\rc-server-2_out.txt 2>&1
scripts/win\rc-server-2.bat-9-claude remote-control --name "Development" --spawn worktree --capacity 4 >> data\rc-server-2_out.txt 2>&1
--
scripts/win\rc-server-3.bat-1-@echo off
scripts/win\rc-server-3.bat-2-REM Claude Code Remote Control — Server 3 (Research)
scripts/win\rc-server-3.bat-3-cd /d Q:\finance-analyzer
scripts/win\rc-server-3.bat:4:set CLAUDECODE=
scripts/win\rc-server-3.bat:5:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\rc-server-3.bat-6-
scripts/win\rc-server-3.bat-7-:restart
scripts/win\rc-server-3.bat-8-echo [%date% %time%] Starting RC server 3 (Research)... >> data\rc-server-3_out.txt 2>&1
scripts/win\rc-server-3.bat-9-claude remote-control --name "Research" --spawn worktree --capacity 4 >> data\rc-server-3_out.txt 2>&1
--
scripts/win\rc-server-ensure.ps1-39-}
scripts/win\rc-server-ensure.ps1-40-
scripts/win\rc-server-ensure.ps1-41-function Send-Telegram($msg) {
scripts/win\rc-server-ensure.ps1-42-    try {
scripts/win\rc-server-ensure.ps1:43:        $cfg = Get-Content $configFile -Raw | ConvertFrom-Json
scripts/win\rc-server-ensure.ps1-44-        $token  = $cfg.telegram.token
scripts/win\rc-server-ensure.ps1-45-        $chatId = $cfg.telegram.chat_id
scripts/win\rc-server-ensure.ps1-46-        if (-not $token -or -not $chatId) { return }
scripts/win\rc-server-ensure.ps1-47-        $body = @{ chat_id = $chatId; text = $msg; parse_mode = "Markdown" } | ConvertTo-Json
scripts/win\rc-server-ensure.ps1:48:        Invoke-RestMethod -Uri "https://api.telegram.org/bot$token/sendMessage" `
scripts/win\rc-server-ensure.ps1-49-            -Method Post -ContentType "application/json" -Body $body `
scripts/win\rc-server-ensure.ps1-50-            -TimeoutSec 15 | Out-Null
scripts/win\rc-server-ensure.ps1-51-    } catch {
scripts/win\rc-server-ensure.ps1-52-        Log "Telegram send failed: $_"
--
scripts/win\rc-server-ensure.ps1-89-    }
scripts/win\rc-server-ensure.ps1-90-    return @{ Alive = $false; Method = "tcp"; Detail = "no connection and no log file" }
scripts/win\rc-server-ensure.ps1-91-}
scripts/win\rc-server-ensure.ps1-92-
scripts/win\rc-server-ensure.ps1:93:# Also find bat-loop cmd.exe processes (parents of the claude.exe servers)
scripts/win\rc-server-ensure.ps1:94:$batProcs = Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" -ErrorAction SilentlyContinue |
scripts/win\rc-server-ensure.ps1-95-    Where-Object { $_.CommandLine -match 'rc-server' }
scripts/win\rc-server-ensure.ps1-96-
scripts/win\rc-server-ensure.ps1-97-$launched = 0
scripts/win\rc-server-ensure.ps1-98-$skipped  = 0
--
scripts/win\rc-server-ensure.ps1-118-            $killed++
scripts/win\rc-server-ensure.ps1-119-            Send-Telegram "*RC Ensure* $($srv.Name) was dead after ${ageHrs}h ($($check.Detail)) -- recycled. Check if any spawned work was interrupted."
scripts/win\rc-server-ensure.ps1-120-            if (-not $batRunning) {
scripts/win\rc-server-ensure.ps1-121-                Log "$($srv.Name) bat loop also missing. Launching $($srv.Bat)..."
scripts/win\rc-server-ensure.ps1:122:                Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$($srv.Bat)`"" -WindowStyle Minimized
scripts/win\rc-server-ensure.ps1-123-                $launched++
scripts/win\rc-server-ensure.ps1-124-            }
scripts/win\rc-server-ensure.ps1-125-        }
scripts/win\rc-server-ensure.ps1-126-    } else {
--
scripts/win\rc-server-ensure.ps1-129-            Log "$($srv.Name) claude.exe gone but bat loop alive (PID $($batRunning.ProcessId)). Will auto-restart."
scripts/win\rc-server-ensure.ps1-130-            $skipped++
scripts/win\rc-server-ensure.ps1-131-        } else {
scripts/win\rc-server-ensure.ps1-132-            Log "$($srv.Name) not running. Launching $($srv.Bat)..."
scripts/win\rc-server-ensure.ps1:133:            Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$($srv.Bat)`"" -WindowStyle Minimized
scripts/win\rc-server-ensure.ps1-134-            $launched++
scripts/win\rc-server-ensure.ps1-135-        }
scripts/win\rc-server-ensure.ps1-136-    }
scripts/win\rc-server-ensure.ps1-137-}
--
scripts/win\rc-server.bat-4-REM Use rc-server-ensure.ps1 to launch all 3 servers independently.
scripts/win\rc-server.bat-5-cd /d Q:\finance-analyzer
scripts/win\rc-server.bat-6-
scripts/win\rc-server.bat-7-REM Clear Claude Code session markers (prevents nested-session errors)
scripts/win\rc-server.bat:8:set CLAUDECODE=
scripts/win\rc-server.bat:9:set CLAUDE_CODE_ENTRYPOINT=
scripts/win\rc-server.bat-10-
scripts/win\rc-server.bat-11-:restart
scripts/win\rc-server.bat-12-echo [%date% %time%] Starting RC server 1 (Trading)... >> data\rc-server_out.txt 2>&1
scripts/win\rc-server.bat-13-claude remote-control --name "Trading" --spawn worktree --capacity 4 >> data\rc-server_out.txt 2>&1
--
scripts/win\rc-watchdog.ps1-8-param(
scripts/win\rc-watchdog.ps1-9-    [int]$MaxAgeHours = 20
scripts/win\rc-watchdog.ps1-10-)
scripts/win\rc-watchdog.ps1-11-
scripts/win\rc-watchdog.ps1:12:$ErrorActionPreference = "Continue"
scripts/win\rc-watchdog.ps1-13-$logFile  = "Q:\finance-analyzer\data\rc-watchdog.log"
scripts/win\rc-watchdog.ps1-14-$configFile = "C:\Users\Herc2\.config\finance-analyzer\config.json"
scripts/win\rc-watchdog.ps1-15-
scripts/win\rc-watchdog.ps1-16-$servers = @(
--
scripts/win\rc-watchdog.ps1-27-}
scripts/win\rc-watchdog.ps1-28-
scripts/win\rc-watchdog.ps1-29-function Send-Telegram($msg) {
scripts/win\rc-watchdog.ps1-30-    try {
scripts/win\rc-watchdog.ps1:31:        $cfg = Get-Content $configFile -Raw | ConvertFrom-Json
scripts/win\rc-watchdog.ps1-32-        $token  = $cfg.telegram.token
scripts/win\rc-watchdog.ps1-33-        $chatId = $cfg.telegram.chat_id
scripts/win\rc-watchdog.ps1-34-        if (-not $token -or -not $chatId) {
scripts/win\rc-watchdog.ps1-35-            Log "Telegram: missing token or chat_id in config"
scripts/win\rc-watchdog.ps1-36-            return
scripts/win\rc-watchdog.ps1-37-        }
scripts/win\rc-watchdog.ps1-38-        $body = @{ chat_id = $chatId; text = $msg; parse_mode = "Markdown" } | ConvertTo-Json
scripts/win\rc-watchdog.ps1:39:        Invoke-RestMethod -Uri "https://api.telegram.org/bot$token/sendMessage" `
scripts/win\rc-watchdog.ps1-40-            -Method Post -ContentType "application/json" -Body $body `
scripts/win\rc-watchdog.ps1-41-            -TimeoutSec 15 | Out-Null
scripts/win\rc-watchdog.ps1-42-        Log "Telegram alert sent"
scripts/win\rc-watchdog.ps1-43-    } catch {
--
scripts/win\rc-watchdog.ps1-50-    Where-Object { $_.CommandLine -match 'remote-control' }
scripts/win\rc-watchdog.ps1-51-
scripts/win\rc-watchdog.ps1-52-$tcpConns = Get-NetTCPConnection -State Established -RemotePort 443 -ErrorAction SilentlyContinue
scripts/win\rc-watchdog.ps1-53-
scripts/win\rc-watchdog.ps1:54:$batProcs = Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" -ErrorAction SilentlyContinue |
scripts/win\rc-watchdog.ps1-55-    Where-Object { $_.CommandLine -match 'rc-server' }
scripts/win\rc-watchdog.ps1-56-
scripts/win\rc-watchdog.ps1-57-function Test-Connected($procId) {
scripts/win\rc-watchdog.ps1-58-    $conn = $tcpConns | Where-Object { $_.OwningProcess -eq $procId }
--
scripts/win\rc-watchdog.ps1-76-            Log "$($srv.Name): claude.exe gone, bat loop alive - will auto-restart"
scripts/win\rc-watchdog.ps1-77-            $healthy++
scripts/win\rc-watchdog.ps1-78-        } else {
scripts/win\rc-watchdog.ps1-79-            Log "$($srv.Name): not running. Launching $($srv.BatFile)..."
scripts/win\rc-watchdog.ps1:80:            Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$basePath\$($srv.BatFile)`"" -WindowStyle Minimized
scripts/win\rc-watchdog.ps1-81-            $missing++
scripts/win\rc-watchdog.ps1-82-            $actions += "$($srv.Name): launched - was missing"
scripts/win\rc-watchdog.ps1-83-        }
scripts/win\rc-watchdog.ps1-84-        continue
--
scripts/win\rc-watchdog.ps1-97-        $msg = "$($srv.Name): recycled at $ageStr, limit ${MaxAgeHours}h"
scripts/win\rc-watchdog.ps1-98-
scripts/win\rc-watchdog.ps1-99-        if (-not $bat) {
scripts/win\rc-watchdog.ps1-100-            Log "$($srv.Name): bat loop missing after recycle. Launching..."
scripts/win\rc-watchdog.ps1:101:            Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$basePath\$($srv.BatFile)`"" -WindowStyle Minimized
scripts/win\rc-watchdog.ps1-102-            $msg += " + relaunched bat"
scripts/win\rc-watchdog.ps1-103-        }
scripts/win\rc-watchdog.ps1-104-        $actions += $msg
scripts/win\rc-watchdog.ps1-105-        continue
--
scripts/win\rc-watchdog.ps1-112-        $zombied++
scripts/win\rc-watchdog.ps1-113-        $msg = "$($srv.Name): killed zombie at $ageStr, no connection"
scripts/win\rc-watchdog.ps1-114-
scripts/win\rc-watchdog.ps1-115-        if (-not $bat) {
scripts/win\rc-watchdog.ps1:116:            Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$basePath\$($srv.BatFile)`"" -WindowStyle Minimized
scripts/win\rc-watchdog.ps1-117-            $msg += " + relaunched bat"
scripts/win\rc-watchdog.ps1-118-        }
scripts/win\rc-watchdog.ps1-119-        $actions += $msg
scripts/win\rc-watchdog.ps1-120-        continue
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

2026-05-11T15:32:29.813706Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: scripts/win/*.bat: The filename, directory name, or volume label syntax is incorrect. (os error 123)
rg: scripts/win/*.ps1: The filename, directory name, or volume label syntax is incorrect. (os error 123)
rg: scripts/*.bat: The filename, directory name, or volume label syntax is incorrect. (os error 123)

 exited 1 in 768ms:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: scripts/win/*.bat: The filename, directory name, or volume label syntax is incorrect. (os error 123)
rg: scripts/win/*.ps1: The filename, directory name, or volume label syntax is incorrect. (os error 123)
rg: scripts/*.bat: The filename, directory name, or volume label syntax is incorrect. (os error 123)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 2 \"python\\.exe|\\.venv\\\\Scripts\\\\python\\.exe|claude |remote-control\" scripts/win" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -l "CLAUDECODE" scripts/win' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -l \"python\\.exe|claude |remote-control\" scripts/win" in Q:\finance-analyzer
 succeeded in 604ms:
scripts/win\crypto-loop.bat
scripts/win\oil-loop.bat
scripts/win\pf-loop.bat
scripts/win\pf-agent.bat
scripts/win\adversarial-review.bat
scripts/win\metals-loop.bat
scripts/win\pf-outcome-check.bat
scripts/win\meta-learner-retrain.bat
scripts/win\rc-server-2.bat
scripts/win\rc-server.bat
scripts/win\rc-server-3.bat
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 626ms:
scripts/win\install-health-check-tasks.ps1-2-# Run as Administrator
scripts/win\install-health-check-tasks.ps1-3-
scripts/win\install-health-check-tasks.ps1:4:$pythonExe = "Q:\finance-analyzer\.venv\Scripts\python.exe"
scripts/win\install-health-check-tasks.ps1-5-$script = "Q:\finance-analyzer\scripts\health_check.py"
scripts/win\install-health-check-tasks.ps1-6-$workDir = "Q:\finance-analyzer"
--
scripts/win\adversarial-review.bat-1-@echo off
scripts/win\adversarial-review.bat-2-REM PF-AdversarialReview — Daily dual adversarial review (Codex + Claude)
scripts/win\adversarial-review.bat:3:REM Runs claude code CLI with the full review prompt.
scripts/win\adversarial-review.bat-4-REM Output: data\adversarial_review_out.txt
scripts/win\adversarial-review.bat-5-cd /d Q:\finance-analyzer
--
scripts/win\adversarial-review.bat-9-echo [%date% %time%] Starting adversarial review... >> data\adversarial_review_out.txt 2>&1
scripts/win\adversarial-review.bat-10-
scripts/win\adversarial-review.bat:11:claude -p "Follow /fgl protocol. Run a full dual adversarial review of the finance-analyzer codebase: partition into 8 subsystems (signals-core, orchestration, portfolio-risk, metals-core, avanza-api, signals-modules, data-external, infrastructure), create worktree with empty-baseline branches, run /codex:adversarial-review in background for each subsystem, write your own independent adversarial review, collect codex results, cross-critique in both directions, write synthesis doc. Commit all docs to main, push via Windows git, clean up worktrees. Do NOT ask for approval — follow /fgl rules. Spend your entire context on this." --allowedTools "Edit,Read,Bash,Write,Glob,Grep" --max-turns 80 >> data\adversarial_review_out.txt 2>&1
scripts/win\adversarial-review.bat-12-
scripts/win\adversarial-review.bat-13-echo [%date% %time%] Adversarial review finished (code %ERRORLEVEL%). >> data\adversarial_review_out.txt 2>&1
--
scripts/win\fin-snipe-manager.bat-1-@echo off
scripts/win\fin-snipe-manager.bat-2-cd /d %~dp0\..\..
scripts/win\fin-snipe-manager.bat:3:.venv\Scripts\python.exe -u -m portfolio.fin_snipe_manager %*
--
scripts/win\metals-arm-stop-once.bat-1-@echo off
scripts/win\metals-arm-stop-once.bat-2-cd /d Q:\finance-analyzer
scripts/win\metals-arm-stop-once.bat:3:.venv\Scripts\python.exe data\arm_stop_orders_once.py %*
--
scripts/win\install-fix-agent-task.ps1-6-
scripts/win\install-fix-agent-task.ps1-7-$taskName    = "PF-FixAgentDispatcher"
scripts/win\install-fix-agent-task.ps1:8:$pythonPath  = "Q:\finance-analyzer\.venv\Scripts\python.exe"
scripts/win\install-fix-agent-task.ps1-9-$scriptPath  = "Q:\finance-analyzer\scripts\fix_agent_dispatcher.py"
scripts/win\install-fix-agent-task.ps1-10-$workingDir  = "Q:\finance-analyzer"
--
scripts/win\fin-snipe.bat-1-@echo off
scripts/win\fin-snipe.bat-2-cd /d %~dp0\..\..
scripts/win\fin-snipe.bat:3:.venv\Scripts\python.exe -u -m portfolio.fin_snipe %*
--
scripts/win\crypto-loop.bat-11-set CLAUDE_CODE_ENTRYPOINT=
scripts/win\crypto-loop.bat-12-echo [%date% %time%] Starting crypto loop...
scripts/win\crypto-loop.bat:13:.venv\Scripts\python.exe -u data\crypto_loop.py --loop > data\crypto_loop_out.txt 2>&1
scripts/win\crypto-loop.bat-14-set EXIT_CODE=%ERRORLEVEL%
scripts/win\crypto-loop.bat-15-echo [%date% %time%] Crypto loop exited (code %EXIT_CODE%).
--
scripts/win\install-oil-loop-task.ps1-44-Write-Host "First-run prerequisite: oil warrant catalog must be populated."
scripts/win\install-oil-loop-task.ps1-45-Write-Host "Run a one-shot probe with a live Avanza session to fill the catalog:"
scripts/win\install-oil-loop-task.ps1:46:Write-Host "  .venv\Scripts\python.exe -u data\oil_loop.py --once --debug"
--
scripts/win\meta-learner-retrain.bat-7-
scripts/win\meta-learner-retrain.bat-8-echo [%date% %time%] Starting meta-learner retraining... >> data\meta_learner_retrain_out.txt 2>&1
scripts/win\meta-learner-retrain.bat:9:start /LOW /B .venv\Scripts\python.exe -u portfolio/meta_learner.py >> data\meta_learner_retrain_out.txt 2>&1
scripts/win\meta-learner-retrain.bat-10-echo [%date% %time%] Meta-learner retraining finished (code %ERRORLEVEL%). >> data\meta_learner_retrain_out.txt 2>&1
--
scripts/win\install-loop-health-watchdog-task.ps1-18-}
scripts/win\install-loop-health-watchdog-task.ps1-19-
scripts/win\install-loop-health-watchdog-task.ps1:20:$action = New-ScheduledTaskAction -Execute "Q:\finance-analyzer\.venv\Scripts\python.exe" `
scripts/win\install-loop-health-watchdog-task.ps1-21-    -Argument "$scriptDir\loop_health_watchdog.py" `
scripts/win\install-loop-health-watchdog-task.ps1-22-    -WorkingDirectory "Q:\finance-analyzer"
--
scripts/win\golddigger-loop.bat-5-:restart
scripts/win\golddigger-loop.bat-6-echo [%date% %time%] Starting GoldDigger...
scripts/win\golddigger-loop.bat:7:.venv\Scripts\python.exe -u -m portfolio.golddigger --dry-run >> data\golddigger_out.txt 2>&1
scripts/win\golddigger-loop.bat-8-set EXIT_CODE=%ERRORLEVEL%
scripts/win\golddigger-loop.bat-9-echo [%date% %time%] GoldDigger exited (code %EXIT_CODE%).
--
scripts/win\install-log-rotate-task.ps1-12-
scripts/win\install-log-rotate-task.ps1-13-$TaskName = "PF-LogRotate"
scripts/win\install-log-rotate-task.ps1:14:$pythonExe = "Q:\finance-analyzer\.venv\Scripts\python.exe"
scripts/win\install-log-rotate-task.ps1-15-$workDir   = "Q:\finance-analyzer"
scripts/win\install-log-rotate-task.ps1-16-
--
scripts/win\golddigger.bat-5-:restart
scripts/win\golddigger.bat-6-echo [%date% %time%] Starting GoldDigger...
scripts/win\golddigger.bat:7:.venv\Scripts\python.exe -u -m portfolio.golddigger %* >> data\golddigger_out.txt 2>&1
scripts/win\golddigger.bat-8-echo [%date% %time%] GoldDigger exited (code %ERRORLEVEL%). Restarting in 30s...
scripts/win\golddigger.bat-9-timeout /t 30 /nobreak >nul
--
scripts/win\install-loop-health-report-task.ps1-18-}
scripts/win\install-loop-health-report-task.ps1-19-
scripts/win\install-loop-health-report-task.ps1:20:$action = New-ScheduledTaskAction -Execute "Q:\finance-analyzer\.venv\Scripts\python.exe" `
scripts/win\install-loop-health-report-task.ps1-21-    -Argument "Q:\finance-analyzer\scripts\loop_health_report.py" `
scripts/win\install-loop-health-report-task.ps1-22-    -WorkingDirectory "Q:\finance-analyzer"
--
scripts/win\oil-loop.bat-10-set CLAUDE_CODE_ENTRYPOINT=
scripts/win\oil-loop.bat-11-echo [%date% %time%] Starting oil loop...
scripts/win\oil-loop.bat:12:.venv\Scripts\python.exe -u data\oil_loop.py --loop > data\oil_loop_out.txt 2>&1
scripts/win\oil-loop.bat-13-set EXIT_CODE=%ERRORLEVEL%
scripts/win\oil-loop.bat-14-echo [%date% %time%] Oil loop exited (code %EXIT_CODE%).
--
scripts/win\install-loop-health-daily-task.ps1-29-}
scripts/win\install-loop-health-daily-task.ps1-30-
scripts/win\install-loop-health-daily-task.ps1:31:$action = New-ScheduledTaskAction -Execute "Q:\finance-analyzer\.venv\Scripts\python.exe" `
scripts/win\install-loop-health-daily-task.ps1-32-    -Argument "$scriptDir\loop_health_report.py" `
scripts/win\install-loop-health-daily-task.ps1-33-    -WorkingDirectory "Q:\finance-analyzer"
--
scripts/win\metals-loop.bat-9-set CLAUDE_CODE_ENTRYPOINT=
scripts/win\metals-loop.bat-10-echo [%date% %time%] Starting metals loop...
scripts/win\metals-loop.bat:11:.venv\Scripts\python.exe -u data\metals_loop.py > data\metals_loop_out.txt 2>&1
scripts/win\metals-loop.bat-12-set EXIT_CODE=%ERRORLEVEL%
scripts/win\metals-loop.bat-13-echo [%date% %time%] Metals loop exited (code %EXIT_CODE%).
--
scripts/win\pf-local-llm-report.bat-10-
scripts/win\pf-local-llm-report.bat-11-set LOG_FILE=%REPO_ROOT%\data\local_llm_report_task.log
scripts/win\pf-local-llm-report.bat:12:set PYTHON_EXE=%REPO_ROOT%\.venv\Scripts\python.exe
scripts/win\pf-local-llm-report.bat:13:if not exist "%PYTHON_EXE%" set PYTHON_EXE=Q:\finance-analyzer\.venv\Scripts\python.exe
scripts/win\pf-local-llm-report.bat-14-
scripts/win\pf-local-llm-report.bat-15-if not exist "%PYTHON_EXE%" (
--
scripts/win\mstr-loop.bat-5-
scripts/win\mstr-loop.bat-6-cd /d Q:\finance-analyzer
scripts/win\mstr-loop.bat:7:.venv\Scripts\python.exe -u -m portfolio.mstr_loop >> logs\mstr_loop_out.txt 2>&1
--
scripts/win\pf-agent.bat-14-REM Layer 1 already wrote fresh agent_summary.json — no need to re-collect
scripts/win\pf-agent.bat-15-echo Running trading agent...
scripts/win\pf-agent.bat:16:claude -p "You are the Layer 2 trading agent. FIRST read docs/TRADING_PLAYBOOK.md for trading rules. Then read data/layer2_context.md (your memory from previous invocations). Then read data/agent_summary_compact.json (signals, trigger reasons, timeframes), data/portfolio_state.json (Patient portfolio), and data/portfolio_state_bold.json (Bold portfolio). Follow the playbook to analyze, decide, and act for BOTH strategies independently. Compare your previous theses and prices with current data — were you right? Always write a journal entry and send a Telegram message." --allowedTools "Edit,Read,Bash,Write" --max-turns 40
--
scripts/win\pf-loop.bat-12-set PYTHONPATH=Q:\finance-analyzer
scripts/win\pf-loop.bat-13-echo [%date% %time%] Starting loop...
scripts/win\pf-loop.bat:14:START /B /WAIT .venv\Scripts\python.exe -u portfolio\main.py --loop >> data\loop_out.txt 2>&1
scripts/win\pf-loop.bat-15-set EXIT_CODE=%ERRORLEVEL%
scripts/win\pf-loop.bat-16-echo [%date% %time%] Loop exited (code %EXIT_CODE%).
--
scripts/win\pf-llm-backfill.bat-4-REM Idempotent; rows without elapsed horizons are skipped and retried later.
scripts/win\pf-llm-backfill.bat-5-cd /d Q:\finance-analyzer
scripts/win\pf-llm-backfill.bat:6:.venv\Scripts\python.exe scripts\backfill_llm_outcomes.py >> data\llm_backfill_out.txt 2>&1
scripts/win\pf-llm-backfill.bat:7:.venv\Scripts\python.exe scripts\backfill_sentiment_shadow.py --horizon 1d >> data\llm_backfill_out.txt 2>&1
--
scripts/win\pf-loop-ensure.ps1-12-# Fallback: check if main.py --loop is running
scripts/win\pf-loop-ensure.ps1-13-if (-not $existing) {
scripts/win\pf-loop-ensure.ps1:14:    $pyLoop = Get-WmiObject Win32_Process -Filter "Name='python.exe'" |
scripts/win\pf-loop-ensure.ps1-15-        Where-Object { $_.CommandLine -match 'main\.py.*--loop' }
scripts/win\pf-loop-ensure.ps1-16-    if ($pyLoop) {
--
scripts/win\pf-outcome-check.bat-7-
scripts/win\pf-outcome-check.bat-8-echo [%date% %time%] Starting outcome backfill... >> data\outcome_check_out.txt 2>&1
scripts/win\pf-outcome-check.bat:9:.venv\Scripts\python.exe -u portfolio\main.py --check-outcomes >> data\outcome_check_out.txt 2>&1
scripts/win\pf-outcome-check.bat-10-echo [%date% %time%] Outcome backfill finished (code %ERRORLEVEL%). >> data\outcome_check_out.txt 2>&1
--
scripts/win\pf-restart.bat-2-REM Thin wrapper around pf-restart.ps1 so the loop can be restarted from cmd.exe
scripts/win\pf-restart.bat-3-REM or `cmd.exe /c scripts\win\pf-restart.bat` from WSL. See pf-restart.ps1 for
scripts/win\pf-restart.bat:4:REM the full rationale on why schtasks /end leaves orphan python.exe children.
scripts/win\pf-restart.bat-5-REM
scripts/win\pf-restart.bat-6-REM Usage:
--
scripts/win\pf-restart.ps1-5-.DESCRIPTION
scripts/win\pf-restart.ps1-6-  `schtasks /end <task>` only signals the wrapper batch script - the
scripts/win\pf-restart.ps1:7:  child python.exe spawned by `START /B /WAIT` survives as an orphan,
scripts/win\pf-restart.ps1-8-  keeping the old code in memory and blocking the file-singleton lock
scripts/win\pf-restart.ps1-9-  the new instance tries to acquire (exit code 11). Result: schtasks
--
scripts/win\pf-restart.ps1-12-
scripts/win\pf-restart.ps1-13-  This script does the right thing in one shot:
scripts/win\pf-restart.ps1:14:    1. Finds python.exe processes whose CommandLine matches the loop
scripts/win\pf-restart.ps1-15-       entry-point (portfolio.main --loop or portfolio.metals_loop).
scripts/win\pf-restart.ps1-16-    2. Stop-Process -Force on every matching PID.
--
scripts/win\pf-restart.ps1-53-    # string is treated as a regex (already escaped where literal characters
scripts/win\pf-restart.ps1-54-    # need it).
scripts/win\pf-restart.ps1:55:    $procs = Get-CimInstance Win32_Process -Filter 'name="python.exe"' |
scripts/win\pf-restart.ps1-56-        Where-Object { $_.CommandLine -match $t.Match }
scripts/win\pf-restart.ps1-57-
--
scripts/win\pf-restart.ps1-64-        Start-Sleep -Seconds 2
scripts/win\pf-restart.ps1-65-    } else {
scripts/win\pf-restart.ps1:66:        Write-Host "    no running python.exe matching '$($t.Match)'"
scripts/win\pf-restart.ps1-67-    }
scripts/win\pf-restart.ps1-68-
--
scripts/win\pf-restart.ps1-82-Start-Sleep -Seconds 5
scripts/win\pf-restart.ps1-83-foreach ($t in $targets) {
scripts/win\pf-restart.ps1:84:    $procs = Get-CimInstance Win32_Process -Filter 'name="python.exe"' |
scripts/win\pf-restart.ps1-85-        Where-Object { $_.CommandLine -match $t.Match }
scripts/win\pf-restart.ps1-86-    if ($procs) {
--
scripts/win\pf-restart.ps1-89-        }
scripts/win\pf-restart.ps1-90-    } else {
scripts/win\pf-restart.ps1:91:        Write-Warning "  $($t.Task): no python.exe found yet - task may still be starting"
scripts/win\pf-restart.ps1-92-    }
scripts/win\pf-restart.ps1-93-}
--
scripts/win\pf-shadow-review.bat-5-REM that as a failure so the task history surfaces the alert.
scripts/win\pf-shadow-review.bat-6-cd /d Q:\finance-analyzer
scripts/win\pf-shadow-review.bat:7:.venv\Scripts\python.exe scripts\review_shadow_signals.py >> data\shadow_review_out.txt 2>&1
--
scripts/win\pf.bat-1-@echo off
scripts/win\pf.bat-2-cd /d Q:\finance-analyzer
scripts/win\pf.bat:3:.venv\Scripts\python.exe scripts\pf.py %*
--
scripts/win\rc-keepalive.ps1-65-# Get all claude.exe processes
scripts/win\rc-keepalive.ps1-66-$allClaude = Get-CimInstance Win32_Process -Filter "Name='claude.exe'" -ErrorAction SilentlyContinue
scripts/win\rc-keepalive.ps1:67:$rcProcs = $allClaude | Where-Object { $_.CommandLine -match 'remote-control' }
scripts/win\rc-keepalive.ps1-68-$childProcs = $allClaude | Where-Object { $_.CommandLine -match 'session-id' }
scripts/win\rc-keepalive.ps1-69-
--
scripts/win\rc-server-2.bat-7-:restart
scripts/win\rc-server-2.bat-8-echo [%date% %time%] Starting RC server 2 (Development)... >> data\rc-server-2_out.txt 2>&1
scripts/win\rc-server-2.bat:9:claude remote-control --name "Development" --spawn worktree --capacity 4 >> data\rc-server-2_out.txt 2>&1
scripts/win\rc-server-2.bat-10-echo [%date% %time%] RC server 2 exited (code %ERRORLEVEL%). Restarting in 15s... >> data\rc-server-2_out.txt 2>&1
scripts/win\rc-server-2.bat-11-timeout /t 15 /nobreak >nul
--
scripts/win\rc-server-ensure.ps1-6-#      If the log file hasn't been modified in 10+ min, the server is stale — even if the
scripts/win\rc-server-ensure.ps1-7-#      process is alive and has a TCP socket open. This catches the case where a session
scripts/win\rc-server-ensure.ps1:8:#      looks healthy by TCP but is no longer registered with Anthropic's remote-control API.
scripts/win\rc-server-ensure.ps1-9-#   2. TCP fallback: if the log file doesn't exist yet (first launch), fall back to checking
scripts/win\rc-server-ensure.ps1-10-#      for an ESTABLISHED TCP connection to port 443.
--
scripts/win\rc-server-ensure.ps1-61-}
scripts/win\rc-server-ensure.ps1-62-
scripts/win\rc-server-ensure.ps1:63:# Get all claude remote-control processes once
scripts/win\rc-server-ensure.ps1-64-$rcProcs = Get-CimInstance Win32_Process -Filter "Name='claude.exe'" -ErrorAction SilentlyContinue |
scripts/win\rc-server-ensure.ps1:65:    Where-Object { $_.CommandLine -match 'remote-control' }
scripts/win\rc-server-ensure.ps1-66-
scripts/win\rc-server-ensure.ps1-67-# Get all ESTABLISHED TCP connections to port 443 (Anthropic API) — used as fallback
--
scripts/win\rc-server-3.bat-7-:restart
scripts/win\rc-server-3.bat-8-echo [%date% %time%] Starting RC server 3 (Research)... >> data\rc-server-3_out.txt 2>&1
scripts/win\rc-server-3.bat:9:claude remote-control --name "Research" --spawn worktree --capacity 4 >> data\rc-server-3_out.txt 2>&1
scripts/win\rc-server-3.bat-10-echo [%date% %time%] RC server 3 exited (code %ERRORLEVEL%). Restarting in 15s... >> data\rc-server-3_out.txt 2>&1
scripts/win\rc-server-3.bat-11-timeout /t 15 /nobreak >nul
--
scripts/win\rc-watchdog.ps1-48-# --- Gather state ---
scripts/win\rc-watchdog.ps1-49-$rcProcs = Get-CimInstance Win32_Process -Filter "Name='claude.exe'" -ErrorAction SilentlyContinue |
scripts/win\rc-watchdog.ps1:50:    Where-Object { $_.CommandLine -match 'remote-control' }
scripts/win\rc-watchdog.ps1-51-
scripts/win\rc-watchdog.ps1-52-$tcpConns = Get-NetTCPConnection -State Established -RemotePort 443 -ErrorAction SilentlyContinue
--
scripts/win\train-after-hours.bat-2-REM Scheduled via Windows Task Scheduler at 22:30 CET daily
scripts/win\train-after-hours.bat-3-cd /d Q:\finance-analyzer
scripts/win\train-after-hours.bat:4:.venv\Scripts\python.exe -m portfolio.tinylora_trainer
--
scripts/win\rc-server.bat-11-:restart
scripts/win\rc-server.bat-12-echo [%date% %time%] Starting RC server 1 (Trading)... >> data\rc-server_out.txt 2>&1
scripts/win\rc-server.bat:13:claude remote-control --name "Trading" --spawn worktree --capacity 4 >> data\rc-server_out.txt 2>&1
scripts/win\rc-server.bat-14-echo [%date% %time%] RC server 1 exited (code %ERRORLEVEL%). Restarting in 15s... >> data\rc-server_out.txt 2>&1
scripts/win\rc-server.bat-15-timeout /t 15 /nobreak >nul
--
scripts/win\silver-monitor.bat-5-:restart
scripts/win\silver-monitor.bat-6-echo [%date% %time%] Starting Silver Monitor...
scripts/win\silver-monitor.bat:7:.venv\Scripts\python.exe -u data\silver_monitor.py >> data\silver_monitor_out.txt 2>&1
scripts/win\silver-monitor.bat-8-set EXIT_CODE=%ERRORLEVEL%
scripts/win\silver-monitor.bat-9-echo [%date% %time%] Silver Monitor exited (code %EXIT_CODE%).
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 612ms:
scripts/win\golddigger-loop.bat
scripts/win\adversarial-review.bat
scripts/win\crypto-loop.bat
scripts/win\fin-snipe.bat
scripts/win\fin-snipe-manager.bat
scripts/win\install-fix-agent-task.ps1
scripts/win\install-loop-health-daily-task.ps1
scripts/win\install-log-rotate-task.ps1
scripts/win\install-health-check-tasks.ps1
scripts/win\install-loop-health-report-task.ps1
scripts/win\golddigger.bat
scripts/win\install-loop-health-watchdog-task.ps1
scripts/win\install-oil-loop-task.ps1
scripts/win\metals-loop.bat
scripts/win\oil-loop.bat
scripts/win\metals-arm-stop-once.bat
scripts/win\mstr-loop.bat
scripts/win\meta-learner-retrain.bat
scripts/win\pf-llm-backfill.bat
scripts/win\pf-agent.bat
scripts/win\pf-loop-ensure.ps1
scripts/win\pf-local-llm-report.bat
scripts/win\pf-outcome-check.bat
scripts/win\pf-loop.bat
scripts/win\pf-restart.bat
scripts/win\pf-restart.ps1
scripts/win\pf-shadow-review.bat
scripts/win\pf.bat
scripts/win\rc-server-2.bat
scripts/win\rc-keepalive.ps1
scripts/win\rc-server-3.bat
scripts/win\rc-server-ensure.ps1
scripts/win\rc-server.bat
scripts/win\rc-watchdog.ps1
scripts/win\silver-monitor.bat
scripts/win\train-after-hours.bat
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"kill_orphaned_by_cmdline\\(|kill_orphaned_llama\\(|run_safe\\(|popen_in_job\\(\" portfolio scripts" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 4 \"def _get_collection|_collection is not None|collection_name|collection\\.count\\(\\)|collection\\.get\\(|collection\\.add\\(|_load_journal_entries\\(|with open\\(JOURNAL_FILE\" portfolio/vector_memory.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 4 \"add_shadow\\(|resolve_shadow\\(|save_registry\\(|load_registry\\(|atomic_write_json\\(|load_json\\(\" portfolio/shadow_registry.py" in Q:\finance-analyzer
 succeeded in 737ms:
60-def _now() -> str:
61-    return _dt.datetime.now(_dt.UTC).isoformat()
62-
63-
64:def load_registry(path: Path | str | None = None) -> dict:
65-    """Load the registry. Returns `{"shadows": {}}` when the file is
66-    missing or malformed (never raises)."""
67-    p = Path(path) if path else _REGISTRY_FILE
68:    data = load_json(str(p), default=None)
69-    if not isinstance(data, dict) or "shadows" not in data:
70-        return {"shadows": {}}
71-    return data
72-
73-
74:def save_registry(data: dict, path: Path | str | None = None) -> None:
75-    """Atomically write the registry."""
76-    p = Path(path) if path else _REGISTRY_FILE
77-    p.parent.mkdir(parents=True, exist_ok=True)
78:    atomic_write_json(str(p), data)
79-
80-
81:def add_shadow(
82-    signal: str,
83-    promotion_criteria: dict,
84-    notes: str = "",
85-    *,
--
89-    """Register `signal` as entering shadow. If already present, update
90-    `promotion_criteria` and `notes`, reset status to `"shadow"`, and
91-    refresh `last_reviewed_ts` — but PRESERVE `entered_shadow_ts` so
92-    days-in-shadow accounting survives re-registration."""
93:    reg = load_registry(path=path)
94-    now = _now()
95-    existing = reg["shadows"].get(signal, {})
96-    entered = entered_ts or existing.get("entered_shadow_ts") or now
97-    reg["shadows"][signal] = {
--
100-        "last_reviewed_ts": now,
101-        "status": "shadow",
102-        "notes": notes or existing.get("notes", ""),
103-    }
104:    save_registry(reg, path=path)
105-
106-
107:def resolve_shadow(
108-    signal: str,
109-    status: str,
110-    notes: str = "",
111-    *,
--
114-    """Mark a shadow as promoted or retired. Returns True if found, False
115-    otherwise. Does NOT delete the entry — keeps resolution history."""
116-    if status not in _VALID_STATUS:
117-        raise ValueError(f"status must be in {_VALID_STATUS}, got {status!r}")
118:    reg = load_registry(path=path)
119-    entry = reg["shadows"].get(signal)
120-    if entry is None:
121-        return False
122-    entry["status"] = status
123-    entry["last_reviewed_ts"] = _now()
124-    if notes:
125-        entry["notes"] = notes
126:    save_registry(reg, path=path)
127-    return True
128-
129-
130-def days_in_shadow(signal: str, *, path: Path | str | None = None,
131-                    now: _dt.datetime | None = None) -> float | None:
132-    """Return days elapsed since signal entered shadow. None if unknown."""
133:    reg = load_registry(path=path)
134-    entry = reg["shadows"].get(signal)
135-    if entry is None:
136-        return None
137-    entered_raw = entry.get("entered_shadow_ts")
--
152-                   now: _dt.datetime | None = None) -> list[dict]:
153-    """Return shadow entries that are still `"shadow"` and older than
154-    `stale_days`. Each dict includes `signal`, `days_in_shadow`, and the
155-    full entry for convenience."""
156:    reg = load_registry(path=path)
157-    stale = []
158-    for sig, entry in reg["shadows"].items():
159-        if entry.get("status") != "shadow":
160-            continue
--
172-
173-def seed_defaults(path: Path | str | None = None) -> None:
174-    """Idempotent one-time seeding for the 2026-04-21 LLM-health audit.
175-    Only adds entries that don't already exist — safe to re-run."""
176:    reg = load_registry(path=path)
177-    defaults = {
178-        "fingpt": {
179-            "entered_shadow_ts": "2026-04-09T00:00:00+00:00",
180-            "promotion_criteria": {
--
235-            **cfg,
236-            "last_reviewed_ts": _now(),
237-            "status": "shadow",
238-        }
239:    save_registry(reg, path=path)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 770ms:
portfolio\ministral_trader.py:49:    proc = run_safe(
portfolio\ministral_signal.py:88:    result = run_safe(
portfolio\ministral_signal.py:106:        killed = kill_orphaned_llama()
portfolio\qwen3_trader.py:52:    proc = run_safe(
portfolio\qwen3_signal.py:102:        result = run_safe(
portfolio\qwen3_signal.py:141:    result = run_safe(
portfolio\qwen3_signal.py:166:        killed = kill_orphaned_llama()
portfolio\subprocess_utils.py:4:- run_safe(): Drop-in subprocess.run() replacement that uses Windows Job Objects
portfolio\subprocess_utils.py:6:- popen_in_job(): Popen wrapper for long-running subprocesses — assigns the child
portfolio\subprocess_utils.py:8:- kill_orphaned_by_cmdline(): Find and kill orphaned processes matching a command
portfolio\subprocess_utils.py:10:- kill_orphaned_llama(): Safety-net reaper for orphaned llama-completion.exe processes.
portfolio\subprocess_utils.py:21:def run_safe(cmd, **kwargs):
portfolio\subprocess_utils.py:156:def popen_in_job(cmd, **kwargs):
portfolio\subprocess_utils.py:198:def kill_orphaned_by_cmdline(pattern, exclude_pid=None):
portfolio\subprocess_utils.py:250:def kill_orphaned_llama():
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 772ms:
30-_collection = None
31-_client = None
32-
33-
34:def _get_collection(collection_name="trade_journal"):
35-    """Lazy-init ChromaDB client and return the collection.
36-
37-    Raises ImportError if chromadb is not installed.
38-    """
39-    global _client, _collection
40-
41:    if _collection is not None:
42-        return _collection
43-
44-    import chromadb
45-
46-    CHROMADB_DIR.mkdir(parents=True, exist_ok=True)
47-    _client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
48-    _collection = _client.get_or_create_collection(
49:        name=collection_name,
50-        metadata={"hnsw:space": "cosine"},
51-    )
52-    logger.info("ChromaDB collection '%s' ready (%d entries)",
53:                collection_name, _collection.count())
54-    return _collection
55-
56-
57-def entry_to_text(entry):
--
113-    ts = entry.get("ts", "")
114-    return hashlib.md5(ts.encode()).hexdigest()
115-
116-
117:def embed_entries(entries, collection_name="trade_journal"):
118-    """Embed journal entries that aren't yet in ChromaDB.
119-
120-    Args:
121-        entries: list of journal entry dicts.
122:        collection_name: ChromaDB collection name.
123-
124-    Returns:
125-        int: number of newly embedded entries.
126-    """
127:    collection = _get_collection(collection_name)
128-
129-    # Get existing IDs
130-    existing = set()
131:    if collection.count() > 0:
132:        result = collection.get()
133-        existing = set(result["ids"])
134-
135-    new_docs = []
136-    new_ids = []
--
150-            "regime": entry.get("regime", ""),
151-        })
152-
153-    if new_docs:
154:        collection.add(documents=new_docs, ids=new_ids, metadatas=new_metas)
155-        logger.info("Embedded %d new journal entries", len(new_docs))
156-
157-    return len(new_docs)
158-
159-
160:def query_similar(query_text, top_k=5, collection_name="trade_journal"):
161-    """Query ChromaDB for journal entries similar to query_text.
162-
163-    Args:
164-        query_text: text describing current market state.
165-        top_k: number of results.
166:        collection_name: ChromaDB collection name.
167-
168-    Returns:
169-        list of dicts with keys: text, ts, regime, distance.
170-    """
171:    collection = _get_collection(collection_name)
172:    if collection.count() == 0:
173-        return []
174-
175-    results = collection.query(
176-        query_texts=[query_text],
177:        n_results=min(top_k, collection.count()),
178-    )
179-
180-    entries = []
181-    for i, doc in enumerate(results["documents"][0]):
--
222-    return "\n".join(parts) if parts else ""
223-
224-
225-def get_semantic_context(market_state, bm25_timestamps=None,
226:                         top_k=5, collection_name="trade_journal"):
227-    """Full semantic retrieval pipeline: embed new entries, query, de-dup.
228-
229-    Args:
230-        market_state: dict with signals, held_tickers, regime, prices.
231-        bm25_timestamps: set of timestamp strings already returned by BM25
232-            (for de-duplication).
233-        top_k: number of semantic results to return.
234:        collection_name: ChromaDB collection name.
235-
236-    Returns:
237-        list of dicts with text, ts, regime, distance.
238-        Returns empty list on any error.
239-    """
240-    try:
241-        # Embed any new entries
242:        entries = _load_journal_entries()
243-        if entries:
244:            embed_entries(entries, collection_name)
245-
246-        # Build query
247-        query = build_query_text(market_state)
248-        if not query:
249-            return []
250-
251-        # Query
252:        results = query_similar(query, top_k=top_k * 2, collection_name=collection_name)
253-
254-        # De-duplicate against BM25
255-        bm25_ts = set(bm25_timestamps or [])
256-        deduped = [r for r in results if r.get("ts", "") not in bm25_ts]
--
263-        logger.warning("vector memory error: %s", e)
264-        return []
265-
266-
267:def _load_journal_entries():
268-    """Load all journal entries from JSONL."""
269-    if not JOURNAL_FILE.exists():
270-        return []
271-    entries = []
272:    with open(JOURNAL_FILE, encoding="utf-8") as f:
273-        for line in f:
274-            line = line.strip()
275-            if not line:
276-                continue
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"_get_config\\(|dashboard_token|telegram\\.token|api_key|secret|chat_id|Authorization|Bearer|Cf-Access\" dashboard/app.py dashboard/auth.py dashboard/house_blueprint.py" in Q:\finance-analyzer
 succeeded in 687ms:
dashboard/house_blueprint.py:6:protected by the same `pf_dashboard_token` that gates the finance dashboard.
dashboard/house_blueprint.py:58:    cfg = _get_config()
dashboard/auth.py:18:Tests that patched `dashboard.app._get_dashboard_token` or
dashboard/auth.py:45:COOKIE_NAME = "pf_dashboard_token"
dashboard/auth.py:69:def _get_config() -> dict:
dashboard/auth.py:81:def _get_dashboard_token() -> str | None:
dashboard/auth.py:82:    """Return the configured dashboard_token, or None if not set."""
dashboard/auth.py:83:    return _get_config().get("dashboard_token") or None
dashboard/auth.py:107:      0. Cf-Access-Authenticated-User-Email header — Cloudflare Access has
dashboard/auth.py:109:      1. Cookie (`pf_dashboard_token`) — for repeat visits.
dashboard/auth.py:111:      3. Authorization: Bearer header — for CLI / script clients.
dashboard/auth.py:116:    If no dashboard_token is configured, access is allowed (backwards
dashboard/auth.py:121:        expected = _get_dashboard_token()
dashboard/auth.py:126:        # Cf-Access-* headers at its edge and re-injects them only after
dashboard/auth.py:132:        cf_email = request.headers.get("Cf-Access-Authenticated-User-Email")
dashboard/auth.py:133:        cf_jwt = request.headers.get("Cf-Access-Jwt-Assertion")
dashboard/auth.py:147:        # 3. Authorization: Bearer (CLI / script clients — no cookie set
dashboard/auth.py:149:        auth_header = request.headers.get("Authorization", "")
dashboard/auth.py:150:        if auth_header.startswith("Bearer "):
dashboard/auth.py:159:                "cookie. Replace YOUR_TOKEN with the dashboard_token from "
dashboard/app.py:58:    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
dashboard/app.py:171:def _get_config():
dashboard/app.py:301:    cfg = (_get_config() or {}).get("golddigger", {})
dashboard/app.py:745:    _get_dashboard_token,
dashboard/app.py:779:    """Clear the pf_dashboard_token cookie and redirect to /.
dashboard/app.py:794:        "pf_dashboard_token",
dashboard/app.py:2044:# (data/findapartments runs + innerstad heatmap). Reuses pf_dashboard_token
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 3 \"api_iskbets|iskbets_config|return jsonify\\(\\{\\\"config\\\"\" dashboard/app.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 4 \"Read,Edit,Bash|fix_agent\\.disabled|blocked_until_global|BACKOFF_SCHEDULE_S|consecutive_failures|invoke_claude_fn|allowed_tools\" scripts/fix_agent_dispatcher.py" in Q:\finance-analyzer
2026-05-11T15:34:16.603239Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:api_iskbets|iskbets_config|return jsonify\(\{\)
    ^
error: unclosed group

 exited 1 in 769ms:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:api_iskbets|iskbets_config|return jsonify\(\{\)
    ^
error: unclosed group

 succeeded in 763ms:
37-
38-DATA_DIR = BASE_DIR / "data"
39-CRITICAL_ERRORS_LOG = DATA_DIR / "critical_errors.jsonl"
40-STATE_FILE = DATA_DIR / "fix_agent_state.json"
41:KILL_SWITCH = DATA_DIR / "fix_agent.disabled"
42-
43-# --- Tunables ---
44-# Same constant the loop contract uses so behavior is coherent.
45-SELF_HEAL_COOLDOWN_S = 1800  # 30 min between attempts per category
46-# Exponential backoff on consecutive failures.
47:BACKOFF_SCHEDULE_S = [1800, 7200, 43200]  # 30m → 2h → 12h, then disabled
48-# Recursion guard: dispatcher must refuse to fire if invoked from within
49-# another fix-agent subprocess (env flag propagates to child Claude).
50-RECURSION_ENV = "PF_FIX_AGENT_DEPTH"
51-MAX_RECURSION_DEPTH = 1
--
54-# Per-attempt budget (seconds, Opus 30 turns is typically <15 min).
55-AGENT_TIMEOUT_S = 900
56-AGENT_MAX_TURNS = 30
57-AGENT_MODEL = "opus"
58:AGENT_ALLOWED_TOOLS = "Read,Edit,Bash"
59-
60-logger = logging.getLogger("fix_agent_dispatcher")
61-
62-
--
130-            })
131-        except Exception:
132-            pass
133-        # Conservative default: block ALL spawns for 1 hour via the
134:        # global blocked_until_global field. check_gates honours this
135-        # before any per-category logic.
136-        return {
137-            "by_category": {},
138-            "recursion_counter": 0,
139:            "blocked_until_global": (_now() + timedelta(hours=1)).isoformat(),
140-            "_corrupt_loaded_at": _now().isoformat(),
141-        }
142-
143-
--
235-    if KILL_SWITCH.exists():
236-        return GateDecision(False, "disabled_by_kill_switch")
237-
238-    # Global block — set by _load_state when the state file was corrupt.
239:    global_block = _parse_iso(state.get("blocked_until_global"))
240-    if global_block and global_block > now:
241-        return GateDecision(False, "global_cooldown")
242-
243-    if caller_depth is None:
--
255-
256-def update_state_after_attempt(
257-    state: dict, category: str, success: bool, now: datetime | None = None,
258-) -> dict:
259:    """Bump cooldown + consecutive_failures for the category."""
260-    now = now or _now()
261-    cats = state.setdefault("by_category", {})
262:    entry = cats.setdefault(category, {"consecutive_failures": 0})
263-
264-    entry["last_attempt_ts"] = now.isoformat()
265-    entry["last_attempt_success"] = success
266-
267-    if success:
268:        entry["consecutive_failures"] = 0
269-        entry["blocked_until"] = (now + timedelta(seconds=SELF_HEAL_COOLDOWN_S)).isoformat()
270-    else:
271:        prev = entry.get("consecutive_failures", 0)
272-        new_count = prev + 1
273:        entry["consecutive_failures"] = new_count
274:        idx = min(new_count - 1, len(BACKOFF_SCHEDULE_S) - 1)
275:        if new_count > len(BACKOFF_SCHEDULE_S):
276-            # Beyond the schedule: effectively disabled for 10 years. User
277-            # must manually reset by editing state file or adding a
278-            # resolution line.
279-            entry["blocked_until"] = (now + timedelta(days=3650)).isoformat()
280-        else:
281:            entry["blocked_until"] = (now + timedelta(seconds=BACKOFF_SCHEDULE_S[idx])).isoformat()
282-    return state
283-
284-
285-# ---------------------------------------------------------------------------
--
332-
333-def run(
334-    dry_run: bool = False,
335-    lookback_h: int = LOOKBACK_H,
336:    invoke_claude_fn=None,
337-) -> int:
338-    """Dispatcher entry point. Returns 0 on success (including no-op).
339-
340:    ``invoke_claude_fn`` is dependency-injected for tests; production
341-    passes None and we import at call time.
342-    """
343-    entries = _read_journal(CRITICAL_ERRORS_LOG)
344-    unresolved = _find_unresolved(entries, lookback_h)
--
394-            continue
395-
396-        prompt = build_fix_prompt(category, cat_entries)
397-
398:        if invoke_claude_fn is None:
399:            from portfolio.claude_gate import invoke_claude as invoke_claude_fn  # type: ignore
400-
401-        # Set the recursion env flag only for the duration of the invoke_claude
402-        # call so the child Claude subprocess inherits depth+1 (blocking any
403-        # transitive dispatcher re-entry). Restore afterwards so subsequent
404-        # categories in this same run aren't mistaken for recursion.
405-        prior_env = os.environ.get(RECURSION_ENV)
406-        os.environ[RECURSION_ENV] = str(caller_recursion_depth + 1)
407-        try:
408:            success, exit_code = invoke_claude_fn(
409-                prompt=prompt,
410-                caller=f"fix_agent_{category}",
411-                model=AGENT_MODEL,
412-                max_turns=AGENT_MAX_TURNS,
413:                allowed_tools=AGENT_ALLOWED_TOOLS,
414-                timeout=AGENT_TIMEOUT_S,
415-            )
416-        except Exception as e:
417-            logger.exception("Fix agent invocation raised: %s", e)
--
435-                f"(exit={exit_code}). Check journal for the agent's resolution line."
436-            ),
437-            "context": {"target_category": category, "success": success,
438-                        "exit_code": exit_code,
439:                        "consecutive_failures":
440:                            state["by_category"][category]["consecutive_failures"]},
441-        })
442-
443-    _save_state(state)
444-    logger.info("Dispatcher run complete (spawned=%s)", any_spawned)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 6 'api_iskbets|iskbets_config|return jsonify\\(\\{\"config\"' dashboard/app.py" in Q:\finance-analyzer
 succeeded in 609ms:
1043-        logger.exception("accuracy endpoint error")
1044-        return jsonify({"error": "Internal server error"}), 500
1045-
1046-
1047-@app.route("/api/iskbets")
1048-@require_auth
1049:def api_iskbets():
1050:    config = _read_json(DATA_DIR / "iskbets_config.json")
1051-    state = _read_json(DATA_DIR / "iskbets_state.json")
1052:    return jsonify({"config": config, "state": state})
1053-
1054-
1055-@app.route("/api/lora-status")
1056-@require_auth
1057-def api_lora_status():
1058-    state = _read_json(TRAINING_DIR / "state.json")
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 3 'ExecutionPolicy Bypass|New-ScheduledTaskAction|Get-Content .*ConvertFrom-Json|Invoke-RestMethod -Uri \"https://api.telegram.org/bot\\"'$token/sendMessage"|'"\\"'$configFile ='"' scripts/win/install-rc-server-task.ps1 scripts/win/install-rc-keepalive-task.ps1 scripts/win/install-rc-watchdog-task.ps1 scripts/win/install-loop-resume-task.ps1 scripts/win/rc-watchdog.ps1 scripts/win/rc-keepalive.ps1 scripts/win/rc-server-ensure.ps1" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 3 'New-ScheduledTaskAction|pythonPath|scriptPath|WorkingDirectory|fix_agent_dispatcher\\.py' scripts/win/install-fix-agent-task.ps1" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 3 'CLAUDECODE|CLAUDE_CODE_ENTRYPOINT|python\\.exe|portfolio\\.golddigger|portfolio\\.mstr_loop|backfill_llm_outcomes|review_shadow_signals|portfolio\\.fin_snipe' scripts/win/golddigger.bat scripts/win/golddigger-loop.bat scripts/win/mstr-loop.bat scripts/win/pf-llm-backfill.bat scripts/win/pf-shadow-review.bat scripts/win/fin-snipe.bat scripts/win/fin-snipe-manager.bat scripts/win/silver-monitor.bat scripts/win/train-after-hours.bat" in Q:\finance-analyzer
 succeeded in 776ms:
1-# install-fix-agent-task.ps1 — Run as Administrator
2-# Creates PF-FixAgentDispatcher: fires every 10 minutes, runs
3:# scripts/fix_agent_dispatcher.py. The dispatcher is a no-op when
4-# data/critical_errors.jsonl has no unresolved entries, so firing
5-# frequently is cheap. See docs/plans/2026-04-13-auto-spawn-fix-agent.md.
6-
7-$taskName    = "PF-FixAgentDispatcher"
8:$pythonPath  = "Q:\finance-analyzer\.venv\Scripts\python.exe"
9:$scriptPath  = "Q:\finance-analyzer\scripts\fix_agent_dispatcher.py"
10-$workingDir  = "Q:\finance-analyzer"
11-
12-# Remove existing task if present (idempotent install)
--
19-    -RepetitionInterval (New-TimeSpan -Minutes 10) `
20-    -RepetitionDuration ([TimeSpan]::MaxValue)
21-
22:# Action: python scripts/fix_agent_dispatcher.py
23:$action = New-ScheduledTaskAction `
24:    -Execute $pythonPath `
25:    -Argument "-u `"$scriptPath`"" `
26:    -WorkingDirectory $workingDir
27-
28-# Settings: cap runtime at 20 minutes (agent timeout is 15 min + buffer).
29-# Don't run if battery, skip if a previous instance is still running (
--
50-
51-Write-Host ""
52-Write-Host "=== $taskName installed ==="
53:Write-Host "Every 10 minutes: $pythonPath -u $scriptPath"
54-Write-Host "Working dir:       $workingDir"
55-Write-Host "Kill switch:       touch Q:\finance-analyzer\data\fix_agent.disabled"
56-Write-Host ""
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 789ms:
scripts/win/install-loop-resume-task.ps1-11-$trigger = New-ScheduledTaskTrigger -AtLogOn  # placeholder, replaced by XML below
scripts/win/install-loop-resume-task.ps1-12-
scripts/win/install-loop-resume-task.ps1-13-# Action: run the ensure script
scripts/win/install-loop-resume-task.ps1:14:$action = New-ScheduledTaskAction `
scripts/win/install-loop-resume-task.ps1-15-    -Execute "powershell.exe" `
scripts/win/install-loop-resume-task.ps1:16:    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
scripts/win/install-loop-resume-task.ps1-17-
scripts/win/install-loop-resume-task.ps1-18-# Settings
scripts/win/install-loop-resume-task.ps1-19-$settings = New-ScheduledTaskSettingsSet `
--
scripts/win/rc-keepalive.ps1-25-)
scripts/win/rc-keepalive.ps1-26-
scripts/win/rc-keepalive.ps1-27-$logFile = "Q:\finance-analyzer\data\rc-keepalive.log"
scripts/win/rc-keepalive.ps1:28:$configFile = "C:\Users\Herc2\.config\finance-analyzer\config.json"
scripts/win/rc-keepalive.ps1-29-
scripts/win/rc-keepalive.ps1-30-# Staggered thresholds per server - keeps recycling naturally offset by ~2 min.
scripts/win/rc-keepalive.ps1-31-# All well under the ~20 min Anthropic TTL (4-8 min margin).
--
scripts/win/rc-keepalive.ps1-45-
scripts/win/rc-keepalive.ps1-46-function Send-Telegram($msg) {
scripts/win/rc-keepalive.ps1-47-    try {
scripts/win/rc-keepalive.ps1:48:        $cfg = Get-Content $configFile -Raw | ConvertFrom-Json
scripts/win/rc-keepalive.ps1-49-        $token  = $cfg.telegram.token
scripts/win/rc-keepalive.ps1-50-        $chatId = $cfg.telegram.chat_id
scripts/win/rc-keepalive.ps1-51-        if (-not $token -or -not $chatId) { return }
scripts/win/rc-keepalive.ps1-52-        $body = @{ chat_id = $chatId; text = $msg; parse_mode = "Markdown" } | ConvertTo-Json
scripts/win/rc-keepalive.ps1:53:        Invoke-RestMethod -Uri "https://api.telegram.org/bot$token/sendMessage" `
scripts/win/rc-keepalive.ps1-54-            -Method Post -ContentType "application/json" -Body $body `
scripts/win/rc-keepalive.ps1-55-            -TimeoutSec 15 | Out-Null
scripts/win/rc-keepalive.ps1-56-    } catch {
--
scripts/win/install-rc-watchdog-task.ps1-9-# Remove existing task if present
scripts/win/install-rc-watchdog-task.ps1-10-Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
scripts/win/install-rc-watchdog-task.ps1-11-
scripts/win/install-rc-watchdog-task.ps1:12:$action = New-ScheduledTaskAction `
scripts/win/install-rc-watchdog-task.ps1-13-    -Execute "powershell.exe" `
scripts/win/install-rc-watchdog-task.ps1:14:    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
scripts/win/install-rc-watchdog-task.ps1-15-
scripts/win/install-rc-watchdog-task.ps1-16-# Trigger: every 30 minutes, starting now, repeating indefinitely
scripts/win/install-rc-watchdog-task.ps1-17-$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `
--
scripts/win/install-rc-server-task.ps1-9-$logonTask = "PF-RemoteControl"
scripts/win/install-rc-server-task.ps1-10-Unregister-ScheduledTask -TaskName $logonTask -Confirm:$false -ErrorAction SilentlyContinue
scripts/win/install-rc-server-task.ps1-11-
scripts/win/install-rc-server-task.ps1:12:$logonAction = New-ScheduledTaskAction `
scripts/win/install-rc-server-task.ps1-13-    -Execute "powershell.exe" `
scripts/win/install-rc-server-task.ps1:14:    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
scripts/win/install-rc-server-task.ps1-15-
scripts/win/install-rc-server-task.ps1-16-$logonTrigger = New-ScheduledTaskTrigger -AtLogOn
scripts/win/install-rc-server-task.ps1-17-
--
scripts/win/install-rc-server-task.ps1-35-$wakeTask = "PF-RemoteControl-Wake"
scripts/win/install-rc-server-task.ps1-36-Unregister-ScheduledTask -TaskName $wakeTask -Confirm:$false -ErrorAction SilentlyContinue
scripts/win/install-rc-server-task.ps1-37-
scripts/win/install-rc-server-task.ps1:38:$wakeAction = New-ScheduledTaskAction `
scripts/win/install-rc-server-task.ps1-39-    -Execute "powershell.exe" `
scripts/win/install-rc-server-task.ps1:40:    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" -WakeDelay"
scripts/win/install-rc-server-task.ps1-41-
scripts/win/install-rc-server-task.ps1-42-# Register with a placeholder trigger first, then replace via XML for event trigger
scripts/win/install-rc-server-task.ps1-43-$placeholderTrigger = New-ScheduledTaskTrigger -AtLogOn
--
scripts/win/rc-server-ensure.ps1-29-    @{ Name = "Research";    Bat = "$basePath\rc-server-3.bat"; Pattern = '--name "?Research'; Log = "$dataPath\rc-server-3_out.txt" }
scripts/win/rc-server-ensure.ps1-30-)
scripts/win/rc-server-ensure.ps1-31-
scripts/win/rc-server-ensure.ps1:32:$configFile = "C:\Users\Herc2\.config\finance-analyzer\config.json"
scripts/win/rc-server-ensure.ps1-33-
scripts/win/rc-server-ensure.ps1-34-function Log($msg) {
scripts/win/rc-server-ensure.ps1-35-    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
--
scripts/win/rc-server-ensure.ps1-40-
scripts/win/rc-server-ensure.ps1-41-function Send-Telegram($msg) {
scripts/win/rc-server-ensure.ps1-42-    try {
scripts/win/rc-server-ensure.ps1:43:        $cfg = Get-Content $configFile -Raw | ConvertFrom-Json
scripts/win/rc-server-ensure.ps1-44-        $token  = $cfg.telegram.token
scripts/win/rc-server-ensure.ps1-45-        $chatId = $cfg.telegram.chat_id
scripts/win/rc-server-ensure.ps1-46-        if (-not $token -or -not $chatId) { return }
scripts/win/rc-server-ensure.ps1-47-        $body = @{ chat_id = $chatId; text = $msg; parse_mode = "Markdown" } | ConvertTo-Json
scripts/win/rc-server-ensure.ps1:48:        Invoke-RestMethod -Uri "https://api.telegram.org/bot$token/sendMessage" `
scripts/win/rc-server-ensure.ps1-49-            -Method Post -ContentType "application/json" -Body $body `
scripts/win/rc-server-ensure.ps1-50-            -TimeoutSec 15 | Out-Null
scripts/win/rc-server-ensure.ps1-51-    } catch {
--
scripts/win/rc-watchdog.ps1-11-
scripts/win/rc-watchdog.ps1-12-$ErrorActionPreference = "Continue"
scripts/win/rc-watchdog.ps1-13-$logFile  = "Q:\finance-analyzer\data\rc-watchdog.log"
scripts/win/rc-watchdog.ps1:14:$configFile = "C:\Users\Herc2\.config\finance-analyzer\config.json"
scripts/win/rc-watchdog.ps1-15-
scripts/win/rc-watchdog.ps1-16-$servers = @(
scripts/win/rc-watchdog.ps1-17-    @{ Name = "Trading";     BatFile = "rc-server.bat";   Pattern = '--name "?Trading' },
--
scripts/win/rc-watchdog.ps1-28-
scripts/win/rc-watchdog.ps1-29-function Send-Telegram($msg) {
scripts/win/rc-watchdog.ps1-30-    try {
scripts/win/rc-watchdog.ps1:31:        $cfg = Get-Content $configFile -Raw | ConvertFrom-Json
scripts/win/rc-watchdog.ps1-32-        $token  = $cfg.telegram.token
scripts/win/rc-watchdog.ps1-33-        $chatId = $cfg.telegram.chat_id
scripts/win/rc-watchdog.ps1-34-        if (-not $token -or -not $chatId) {
--
scripts/win/rc-watchdog.ps1-36-            return
scripts/win/rc-watchdog.ps1-37-        }
scripts/win/rc-watchdog.ps1-38-        $body = @{ chat_id = $chatId; text = $msg; parse_mode = "Markdown" } | ConvertTo-Json
scripts/win/rc-watchdog.ps1:39:        Invoke-RestMethod -Uri "https://api.telegram.org/bot$token/sendMessage" `
scripts/win/rc-watchdog.ps1-40-            -Method Post -ContentType "application/json" -Body $body `
scripts/win/rc-watchdog.ps1-41-            -TimeoutSec 15 | Out-Null
scripts/win/rc-watchdog.ps1-42-        Log "Telegram alert sent"
--
scripts/win/install-rc-keepalive-task.ps1-14-# Remove existing task if present
scripts/win/install-rc-keepalive-task.ps1-15-Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
scripts/win/install-rc-keepalive-task.ps1-16-
scripts/win/install-rc-keepalive-task.ps1:17:$actionPeriodic = New-ScheduledTaskAction `
scripts/win/install-rc-keepalive-task.ps1-18-    -Execute "powershell.exe" `
scripts/win/install-rc-keepalive-task.ps1:19:    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
scripts/win/install-rc-keepalive-task.ps1-20-
scripts/win/install-rc-keepalive-task.ps1:21:$actionWake = New-ScheduledTaskAction `
scripts/win/install-rc-keepalive-task.ps1-22-    -Execute "powershell.exe" `
scripts/win/install-rc-keepalive-task.ps1:23:    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" -Wake"
scripts/win/install-rc-keepalive-task.ps1-24-
scripts/win/install-rc-keepalive-task.ps1-25-# Trigger 1: every 5 minutes, starting now, repeating for ~25 years (max safe duration)
scripts/win/install-rc-keepalive-task.ps1-26-$triggerPeriodic = New-ScheduledTaskTrigger -Once -At (Get-Date) `
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 776ms:
scripts/win/train-after-hours.bat-1-@echo off
scripts/win/train-after-hours.bat-2-REM Scheduled via Windows Task Scheduler at 22:30 CET daily
scripts/win/train-after-hours.bat-3-cd /d Q:\finance-analyzer
scripts/win/train-after-hours.bat:4:.venv\Scripts\python.exe -m portfolio.tinylora_trainer
--
scripts/win/silver-monitor.bat-4-
scripts/win/silver-monitor.bat-5-:restart
scripts/win/silver-monitor.bat-6-echo [%date% %time%] Starting Silver Monitor...
scripts/win/silver-monitor.bat:7:.venv\Scripts\python.exe -u data\silver_monitor.py >> data\silver_monitor_out.txt 2>&1
scripts/win/silver-monitor.bat-8-set EXIT_CODE=%ERRORLEVEL%
scripts/win/silver-monitor.bat-9-echo [%date% %time%] Silver Monitor exited (code %EXIT_CODE%).
scripts/win/silver-monitor.bat-10-
--
scripts/win/fin-snipe-manager.bat-1-@echo off
scripts/win/fin-snipe-manager.bat-2-cd /d %~dp0\..\..
scripts/win/fin-snipe-manager.bat:3:.venv\Scripts\python.exe -u -m portfolio.fin_snipe_manager %*
--
scripts/win/fin-snipe.bat-1-@echo off
scripts/win/fin-snipe.bat-2-cd /d %~dp0\..\..
scripts/win/fin-snipe.bat:3:.venv\Scripts\python.exe -u -m portfolio.fin_snipe %*
--
scripts/win/pf-llm-backfill.bat-3-REM Runs the probability-log outcome backfill + sentiment A/B shadow backfill.
scripts/win/pf-llm-backfill.bat-4-REM Idempotent; rows without elapsed horizons are skipped and retried later.
scripts/win/pf-llm-backfill.bat-5-cd /d Q:\finance-analyzer
scripts/win/pf-llm-backfill.bat:6:.venv\Scripts\python.exe scripts\backfill_llm_outcomes.py >> data\llm_backfill_out.txt 2>&1
scripts/win/pf-llm-backfill.bat:7:.venv\Scripts\python.exe scripts\backfill_sentiment_shadow.py --horizon 1d >> data\llm_backfill_out.txt 2>&1
--
scripts/win/pf-shadow-review.bat-4-REM Exit code 1 when stale shadows exist — Windows Task Scheduler treats
scripts/win/pf-shadow-review.bat-5-REM that as a failure so the task history surfaces the alert.
scripts/win/pf-shadow-review.bat-6-cd /d Q:\finance-analyzer
scripts/win/pf-shadow-review.bat:7:.venv\Scripts\python.exe scripts\review_shadow_signals.py >> data\shadow_review_out.txt 2>&1
--
scripts/win/mstr-loop.bat-4-REM Phase is read from MSTR_LOOP_PHASE env var, default "shadow".
scripts/win/mstr-loop.bat-5-
scripts/win/mstr-loop.bat-6-cd /d Q:\finance-analyzer
scripts/win/mstr-loop.bat:7:.venv\Scripts\python.exe -u -m portfolio.mstr_loop >> logs\mstr_loop_out.txt 2>&1
--
scripts/win/golddigger.bat-4-
scripts/win/golddigger.bat-5-:restart
scripts/win/golddigger.bat-6-echo [%date% %time%] Starting GoldDigger...
scripts/win/golddigger.bat:7:.venv\Scripts\python.exe -u -m portfolio.golddigger %* >> data\golddigger_out.txt 2>&1
scripts/win/golddigger.bat-8-echo [%date% %time%] GoldDigger exited (code %ERRORLEVEL%). Restarting in 30s...
scripts/win/golddigger.bat-9-timeout /t 30 /nobreak >nul
scripts/win/golddigger.bat-10-goto restart
--
scripts/win/golddigger-loop.bat-4-
scripts/win/golddigger-loop.bat-5-:restart
scripts/win/golddigger-loop.bat-6-echo [%date% %time%] Starting GoldDigger...
scripts/win/golddigger-loop.bat:7:.venv\Scripts\python.exe -u -m portfolio.golddigger --dry-run >> data\golddigger_out.txt 2>&1
scripts/win/golddigger-loop.bat-8-set EXIT_CODE=%ERRORLEVEL%
scripts/win/golddigger-loop.bat-9-echo [%date% %time%] GoldDigger exited (code %EXIT_CODE%).
scripts/win/golddigger-loop.bat-10-
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 2 'sock\\.bind|IPV6_V6ONLY|ThreadedWSGIServer|serve_forever' dashboard/app.py" in Q:\finance-analyzer
 succeeded in 566ms:
1697-# ---------------------------------------------------------------------------
1698-# Avanza worker thread — Playwright's sync API is bound to its creator
1699:# thread, but Flask's ThreadedWSGIServer spawns a fresh worker per request.
1700-# A request that lands on a different thread than the one which initialised
1701-# Playwright fails with "cannot switch to a different thread (which happens
--
2067-
2068-    Switching to `host="::"` would fix localhost on Linux but on
2069:    Windows the default `IPV6_V6ONLY=True` socket option means IPv4
2070-    clients can no longer connect. So we bind manually with
2071:    `IPV6_V6ONLY=0`, which works on every modern Windows / Linux /
2072-    macOS host.
2073-    """
2074-    import socket
2075:    from werkzeug.serving import ThreadedWSGIServer
2076-
2077:    # Build the dual-stack listening socket explicitly. IPV6_V6ONLY=0
2078-    # enables IPv4 mapping (::ffff:127.0.0.1 etc.), so a single AF_INET6
2079-    # socket accepts both IPv4 and IPv6 clients.
2080-    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
2081:    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
2082-    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
2083:    sock.bind(("::", port))
2084-    sock.listen(128)
2085-
2086:    # ThreadedWSGIServer accepts `fd=` so it skips its own bind/listen
2087-    # and reuses our pre-configured socket. ThreadingMixIn handles
2088-    # concurrent requests just like Werkzeug's default app.run().
2089:    server = ThreadedWSGIServer("::", port, app, fd=sock.fileno())
2090-    print(f"Dashboard listening on dual-stack [::]:{port} (IPv4 + IPv6)",
2091-          flush=True)
2092:    server.serve_forever()
2093-
2094-
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 3 'golddigger\\.bat|schtasks /Create|New-ScheduledTaskAction|cmd\\.exe|/TR ' scripts/win/install-golddigger-task.ps1" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 3 'mstr-loop\\.bat|New-ScheduledTaskAction|cmd\\.exe' scripts/win/install-mstr-loop-task.ps1" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 3 'portfolio\\.golddigger|CLAUDECODE|CLAUDE_CODE_ENTRYPOINT|python\\.exe' scripts/win/golddigger.bat scripts/win/golddigger-loop.bat scripts/win/mstr-loop.bat" in Q:\finance-analyzer
 succeeded in 697ms:
6-$ErrorActionPreference = "Stop"
7-
8-$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
9:$runner = Join-Path $repoRoot "scripts\win\golddigger.bat"
10-
11-if (-not (Test-Path $runner)) {
12-    throw "Runner not found: $runner"
--
19-
20-$taskCommand = '"' + $runner + '" --live'
21-
22:& schtasks /Create /TN $TaskName /SC ONLOGON /TR $taskCommand /F
23-if ($LASTEXITCODE -ne 0) {
24-    exit $LASTEXITCODE
25-}
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 675ms:
18-
19-# We use cmd /c with `set` to inject the phase env var before launching the
20-# wrapper. Going via cmd avoids the PowerShell-vs-batch quoting hell.
21:$action = New-ScheduledTaskAction -Execute "cmd.exe" `
22:    -Argument "/c `"set `"MSTR_LOOP_PHASE=shadow`"&&`"$scriptDir\mstr-loop.bat`"`"" `
23-    -WorkingDirectory "Q:\finance-analyzer"
24-
25-$trigger1 = New-ScheduledTaskTrigger -AtLogOn
--
39-
40-Register-ScheduledTask -TaskName $TaskName `
41-    -Action $action -Trigger $trigger1,$trigger2 -Settings $settings `
42:    -Description "MSTR shadow-mode loop. Runs scripts\win\mstr-loop.bat with PHASE=shadow. Decisions logged to data\mstr_loop_shadow.jsonl, no live orders. Phase A requires 90d shadow data + manual approval."
43-
44-Write-Host "Registered $TaskName (PHASE=shadow)"
45-Write-Host "To start now: Start-ScheduledTask -TaskName '$TaskName'"
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 689ms:
scripts/win/mstr-loop.bat-4-REM Phase is read from MSTR_LOOP_PHASE env var, default "shadow".
scripts/win/mstr-loop.bat-5-
scripts/win/mstr-loop.bat-6-cd /d Q:\finance-analyzer
scripts/win/mstr-loop.bat:7:.venv\Scripts\python.exe -u -m portfolio.mstr_loop >> logs\mstr_loop_out.txt 2>&1
--
scripts/win/golddigger.bat-4-
scripts/win/golddigger.bat-5-:restart
scripts/win/golddigger.bat-6-echo [%date% %time%] Starting GoldDigger...
scripts/win/golddigger.bat:7:.venv\Scripts\python.exe -u -m portfolio.golddigger %* >> data\golddigger_out.txt 2>&1
scripts/win/golddigger.bat-8-echo [%date% %time%] GoldDigger exited (code %ERRORLEVEL%). Restarting in 30s...
scripts/win/golddigger.bat-9-timeout /t 30 /nobreak >nul
scripts/win/golddigger.bat-10-goto restart
--
scripts/win/golddigger-loop.bat-4-
scripts/win/golddigger-loop.bat-5-:restart
scripts/win/golddigger-loop.bat-6-echo [%date% %time%] Starting GoldDigger...
scripts/win/golddigger-loop.bat:7:.venv\Scripts\python.exe -u -m portfolio.golddigger --dry-run >> data\golddigger_out.txt 2>&1
scripts/win/golddigger-loop.bat-8-set EXIT_CODE=%ERRORLEVEL%
scripts/win/golddigger-loop.bat-9-echo [%date% %time%] GoldDigger exited (code %EXIT_CODE%).
scripts/win/golddigger-loop.bat-10-
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 4 'def invoke_claude|CLAUDECODE|CLAUDE_CODE_ENTRYPOINT|os\\.environ|env =' portfolio/claude_gate.py" in Q:\finance-analyzer
 succeeded in 615ms:
93-        return True
94-
95-
96-def _clean_env() -> dict:
97:    """Return a copy of ``os.environ`` with Claude session markers removed.
98-
99-    Prevents the "nested session" error when invoking ``claude -p`` from a
100-    process tree that already has a Claude Code session active.
101-    """
102:    env = os.environ.copy()
103:    env.pop("CLAUDECODE", None)
104:    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
105-    return env
106-
107-
108-# 2026-04-13: Detector for silent auth failures. The `--bare` flag (removed
--
392-# ---------------------------------------------------------------------------
393-# Public API
394-# ---------------------------------------------------------------------------
395-
396:def invoke_claude(
397-    prompt: str,
398-    caller: str,
399-    model: str = "sonnet",
400-    max_turns: int = 20,
--
550-
551-    return status == "invoked", exit_code
552-
553-
554:def invoke_claude_text(
555-    prompt: str,
556-    caller: str,
557-    model: str = "sonnet",
558-    timeout: int = 60,
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -C 3 '_clean_env\\(|subprocess\\.run\\(|subprocess\\.Popen\\(|env=_clean_env|env = _clean_env|env =' portfolio/claude_gate.py" in Q:\finance-analyzer
 succeeded in 616ms:
5-silver_monitor, claude_fundamental, analyze, bigbet, iskbets, etc. — MUST
6-route through ``invoke_claude()`` defined here.
7-
8:Direct ``subprocess.Popen([claude_cmd, "-p", ...])`` calls are FORBIDDEN.
9-Doing so bypasses the kill switch, rate limiter, and invocation tracking.
10-
11-Usage::
--
93-        return True
94-
95-
96:def _clean_env() -> dict:
97-    """Return a copy of ``os.environ`` with Claude session markers removed.
98-
99-    Prevents the "nested session" error when invoking ``claude -p`` from a
100-    process tree that already has a Claude Code session active.
101-    """
102:    env = os.environ.copy()
103-    env.pop("CLAUDECODE", None)
104-    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
105-    return env
--
279-    return count
280-
281-
282:# A-IN-2 (2026-04-11): The previous code used `subprocess.run(timeout=...)`.
283-# CPython's run() does kill the *direct* child on TimeoutExpired, but the
284-# Claude CLI is a Node.js process that spawns its own helpers (MCP servers,
285-# the actual claude API client process, etc.). Killing the direct child
--
310-            # taskkill /T = terminate this PID and all child processes,
311-            # /F = force, /PID = the parent PID. Capture stderr to keep
312-            # logs clean if the process already exited between poll() and here.
313:            res = subprocess.run(
314-                ["taskkill", "/T", "/F", "/PID", str(pid)],
315-                capture_output=True, timeout=5,
316-            )
--
361-    and waits up to 5s for the tree to actually exit before returning.
362-    Logs an error if the tree refused to exit.
363-    """
364:    proc = subprocess.Popen(
365-        cmd,
366-        stdout=subprocess.PIPE,
367-        stderr=subprocess.PIPE,
--
497-            rc, _stdout, _stderr, timed_out = _run_with_tree_kill(
498-                cmd,
499-                timeout=timeout,
500:                env=_clean_env(),
501-                cwd=working_dir,
502-                label=f"claude({caller})",
503-            )
--
600-            rc, stdout, _stderr, timed_out = _run_with_tree_kill(
601-                cmd,
602-                timeout=timeout,
603:                env=_clean_env(),
604-                cwd=str(BASE_DIR),
605-                label=f"claude_text({caller})",
606-            )
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
# Codex adversarial review: infrastructure
## Summary
I found 2 auth blockers and 5 high-severity reliability defects across JSONL rewriting, dispatcher backoff durability, subprocess orphan handling, shadow-registry concurrency, and the 7-day critical-error startup check. Cookie `Secure`/`HttpOnly` flags are correct, and I did not find a stale `process_lock.py` lock bug, a PF-FixAgentDispatcher tool-allowlist drift, or a user-derived shell-injection path.

## P0 — Blockers
- `dashboard/auth.py:60-67, 81-83, 121-123` fails open. If `config.json` is unreadable, malformed, or a broken/permission-denied symlink target, `_read_config_uncached()` returns `{}`, `_get_dashboard_token()` returns `None`, and `require_auth()` grants access to everyone.
- `dashboard/auth.py:132-145, 90-99` trusts the mere presence of `Cf-Access-Authenticated-User-Email` and `Cf-Access-Jwt-Assertion`, never validates the JWT, and then issues a 1-year cookie containing the shared `dashboard_token`. Because `dashboard/app.py:2080-2084` binds the server on all interfaces, any client that can reach port `5055` can spoof those headers and mint a replayable dashboard cookie.

## P1 — High
- `portfolio/file_utils.py:222-227` says all target-file mutations must stay inside `jsonl_sidecar_lock`, but `atomic_write_jsonl()` at `287-305` and `prune_jsonl()` at `349-391` rewrite JSONL without that lock. A concurrent `atomic_append_jsonl()` writer can still lose rows during rewrite, so the append-vs-rotation race is still reproducible through sibling helpers.
- `scripts/fix_agent_dispatcher.py:364-443` only persists cooldown/backoff state once, after the whole category loop. If the task dies after handling an early category, that category’s updated `blocked_until` and `consecutive_failures` are lost and the next scheduler tick can respawn it immediately. `scripts/win/install-fix-agent-task.ps1:28-36` makes this realistic by capping runtime at 20 minutes while each agent attempt can run 15 minutes (`scripts/fix_agent_dispatcher.py:54-58`).
- `scripts/check_critical_errors.py:31-35, 59-60, 81-83` mixes naive and aware datetimes. A naive `ts` parsed by `datetime.fromisoformat()` is compared against an aware UTC cutoff, which raises `TypeError` and kills the rolling 7-day startup check instead of surfacing unresolved critical errors.
- `portfolio/subprocess_utils.py:129-133` and `168-176` start the child before `AssignProcessToJobObject()`. If the parent dies in that window, the child is never attached to the job and survives, violating the module’s orphan-kill guarantee for both `run_safe()` and `popen_in_job()`.
- `portfolio/shadow_registry.py:93-104` and `118-126` do unlocked load-modify-write updates. Two producers adding/resolving different shadows concurrently will last-writer-wins each other’s changes; `atomic_write_json()` preserves file integrity, but not both updates.

## P2 — Medium
- The canonical I/O rule is still violated on live paths: `dashboard/app.py:898-914` (`/api/mstr_loop`), `dashboard/house_blueprint.py:107-109, 287-289, 386-389`, `scripts/check_critical_errors.py:38-50`, `scripts/fix_agent_dispatcher.py:80-92, 110-112`, and `portfolio/vector_memory.py:267-281` all use raw `open()/read_text()/json.load[s]()` instead of `file_utils`. That drops the codebase’s standard UTF-8/OSError/corruption handling and violates the explicit project rule.
- `portfolio/vector_memory.py:41-54, 131-155, 242-245, 267-281` is unbounded on the hot path: the singleton ignores later `collection_name` values, every query rereads the entire journal, and every embed pass calls `collection.get()` to materialize the full existing ID set. As `layer2_journal.jsonl` grows, retrieval latency and memory use scale linearly.
- `dashboard/export_static.py:12-20, 58-81, 105-113` still defaults to exporting authenticated API snapshots into `dashboard/static/api-data/`, which Flask serves without auth. The docstring correctly calls that a leak, but the default code path still recreates it.
- `scripts/win/install-rc-server-task.ps1:12-14, 38-40`, `install-rc-keepalive-task.ps1:17-23`, `install-loop-resume-task.ps1:14-16`, and `install-rc-watchdog-task.ps1:12-14` bake `-ExecutionPolicy Bypass` into recurring task actions. `scripts/win/rc-watchdog.ps1:14`, `rc-keepalive.ps1:28`, and `rc-server-ensure.ps1:32` also hardcode a user-specific config path, making the ops surface policy-agnostic and host-specific.

## P3 — Low
- `dashboard/app.py:777-801` makes logout a GET that mutates auth state by clearing the cookie. I did not find data-mutating GET API endpoints, but `/logout` is still CSRFable and should be POST-only.

## Tests missing
- A dashboard auth test where `config.json` is unreadable/corrupt must assert fail-closed, not fail-open.
- A dashboard auth test must prove spoofed `Cf-Access-*` headers do not bypass local auth or mint a `pf_dashboard_token` cookie.
- `file_utils` needs a concurrent append-vs-`atomic_write_jsonl`/`prune_jsonl` test, not just append-vs-rotation.
- `fix_agent_dispatcher` needs a crash/timeout persistence test that kills the run after the first category and verifies the next run still honors that category’s backoff.
- `check_critical_errors` needs mixed naive/aware timestamp coverage on the rolling 7-day filter.
- `subprocess_utils` needs a regression test or redesign that closes the pre-job-assignment orphan window.
- `shadow_registry` needs concurrent writer coverage.
- `vector_memory` needs per-collection isolation and bounded-growth coverage.
tokens used
241,696
# Codex adversarial review: infrastructure
## Summary
I found 2 auth blockers and 5 high-severity reliability defects across JSONL rewriting, dispatcher backoff durability, subprocess orphan handling, shadow-registry concurrency, and the 7-day critical-error startup check. Cookie `Secure`/`HttpOnly` flags are correct, and I did not find a stale `process_lock.py` lock bug, a PF-FixAgentDispatcher tool-allowlist drift, or a user-derived shell-injection path.

## P0 — Blockers
- `dashboard/auth.py:60-67, 81-83, 121-123` fails open. If `config.json` is unreadable, malformed, or a broken/permission-denied symlink target, `_read_config_uncached()` returns `{}`, `_get_dashboard_token()` returns `None`, and `require_auth()` grants access to everyone.
- `dashboard/auth.py:132-145, 90-99` trusts the mere presence of `Cf-Access-Authenticated-User-Email` and `Cf-Access-Jwt-Assertion`, never validates the JWT, and then issues a 1-year cookie containing the shared `dashboard_token`. Because `dashboard/app.py:2080-2084` binds the server on all interfaces, any client that can reach port `5055` can spoof those headers and mint a replayable dashboard cookie.

## P1 — High
- `portfolio/file_utils.py:222-227` says all target-file mutations must stay inside `jsonl_sidecar_lock`, but `atomic_write_jsonl()` at `287-305` and `prune_jsonl()` at `349-391` rewrite JSONL without that lock. A concurrent `atomic_append_jsonl()` writer can still lose rows during rewrite, so the append-vs-rotation race is still reproducible through sibling helpers.
- `scripts/fix_agent_dispatcher.py:364-443` only persists cooldown/backoff state once, after the whole category loop. If the task dies after handling an early category, that category’s updated `blocked_until` and `consecutive_failures` are lost and the next scheduler tick can respawn it immediately. `scripts/win/install-fix-agent-task.ps1:28-36` makes this realistic by capping runtime at 20 minutes while each agent attempt can run 15 minutes (`scripts/fix_agent_dispatcher.py:54-58`).
- `scripts/check_critical_errors.py:31-35, 59-60, 81-83` mixes naive and aware datetimes. A naive `ts` parsed by `datetime.fromisoformat()` is compared against an aware UTC cutoff, which raises `TypeError` and kills the rolling 7-day startup check instead of surfacing unresolved critical errors.
- `portfolio/subprocess_utils.py:129-133` and `168-176` start the child before `AssignProcessToJobObject()`. If the parent dies in that window, the child is never attached to the job and survives, violating the module’s orphan-kill guarantee for both `run_safe()` and `popen_in_job()`.
- `portfolio/shadow_registry.py:93-104` and `118-126` do unlocked load-modify-write updates. Two producers adding/resolving different shadows concurrently will last-writer-wins each other’s changes; `atomic_write_json()` preserves file integrity, but not both updates.

## P2 — Medium
- The canonical I/O rule is still violated on live paths: `dashboard/app.py:898-914` (`/api/mstr_loop`), `dashboard/house_blueprint.py:107-109, 287-289, 386-389`, `scripts/check_critical_errors.py:38-50`, `scripts/fix_agent_dispatcher.py:80-92, 110-112`, and `portfolio/vector_memory.py:267-281` all use raw `open()/read_text()/json.load[s]()` instead of `file_utils`. That drops the codebase’s standard UTF-8/OSError/corruption handling and violates the explicit project rule.
- `portfolio/vector_memory.py:41-54, 131-155, 242-245, 267-281` is unbounded on the hot path: the singleton ignores later `collection_name` values, every query rereads the entire journal, and every embed pass calls `collection.get()` to materialize the full existing ID set. As `layer2_journal.jsonl` grows, retrieval latency and memory use scale linearly.
- `dashboard/export_static.py:12-20, 58-81, 105-113` still defaults to exporting authenticated API snapshots into `dashboard/static/api-data/`, which Flask serves without auth. The docstring correctly calls that a leak, but the default code path still recreates it.
- `scripts/win/install-rc-server-task.ps1:12-14, 38-40`, `install-rc-keepalive-task.ps1:17-23`, `install-loop-resume-task.ps1:14-16`, and `install-rc-watchdog-task.ps1:12-14` bake `-ExecutionPolicy Bypass` into recurring task actions. `scripts/win/rc-watchdog.ps1:14`, `rc-keepalive.ps1:28`, and `rc-server-ensure.ps1:32` also hardcode a user-specific config path, making the ops surface policy-agnostic and host-specific.

## P3 — Low
- `dashboard/app.py:777-801` makes logout a GET that mutates auth state by clearing the cookie. I did not find data-mutating GET API endpoints, but `/logout` is still CSRFable and should be POST-only.

## Tests missing
- A dashboard auth test where `config.json` is unreadable/corrupt must assert fail-closed, not fail-open.
- A dashboard auth test must prove spoofed `Cf-Access-*` headers do not bypass local auth or mint a `pf_dashboard_token` cookie.
- `file_utils` needs a concurrent append-vs-`atomic_write_jsonl`/`prune_jsonl` test, not just append-vs-rotation.
- `fix_agent_dispatcher` needs a crash/timeout persistence test that kills the run after the first category and verifies the next run still honors that category’s backoff.
- `check_critical_errors` needs mixed naive/aware timestamp coverage on the rolling 7-day filter.
- `subprocess_utils` needs a regression test or redesign that closes the pre-job-assignment orphan window.
- `shadow_registry` needs concurrent writer coverage.
- `vector_memory` needs per-collection isolation and bounded-growth coverage.
