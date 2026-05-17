# PLAN — Full Adversarial Review 2026-05-17

## Goal
End-to-end adversarial review of finance-analyzer. Surface bugs, silent failures,
data-corruption hazards, concurrency races, accuracy-regression risks. **No code
changes** in this session — pure audit. Findings feed future bug-fix worktrees.

## Approach
Partition codebase into 8 subsystems. Spawn one fresh review subagent per
subsystem in parallel (background). Each agent reads the current `main` state
(no diff). While agents run, I do my own independent cross-cutting pass. Then
cross-critique: drop hallucinated/low-quality findings, dedupe across subsystems,
classify P1/P2/P3, write synthesis.

## Subsystem partition

### 1. signals-core — voting engine + accuracy plumbing
signal_engine, signal_registry, signal_weights, signal_utils, signal_db,
signal_postmortem, signal_history, signal_state_since, signal_decay_alert,
accuracy_stats, accuracy_degradation, outcome_tracker, ticker_accuracy,
forecast_accuracy, signal_weight_optimizer, train_signal_weights,
correlation_priors, perception_gate.
Agent: pr-review-toolkit:code-reviewer. → 01_signals_core.md

### 2. orchestration — main loop, Layer 2 dispatch, triggers
main, agent_invocation, trigger, trigger_buffer, autonomous,
multi_agent_layer2, claude_gate, gpu_gate, market_timing, session_calendar,
escalation_gate, escalation_router, reporting.
Agent: pr-review-toolkit:code-reviewer. → 02_orchestration.md

### 3. portfolio-risk — bookkeeping, guards, equity, MC
portfolio_mgr, portfolio_validator, trade_guards, trade_risk_classifier,
trade_validation, risk_management, circuit_breaker, equity_curve,
monte_carlo, monte_carlo_risk, cost_model, alert_budget, exposure_coach,
warrant_portfolio, cumulative_tracker.
Agent: pr-review-toolkit:code-reviewer. → 03_portfolio_risk.md

### 4. metals-core — metals/oil/crypto swing loops, fishing, grid
data/metals_loop, data/metals_swing_trader, data/metals_swing_config,
data/metals_signal_tracker, data/metals_risk, data/metals_shared,
data/metals_execution_engine, data/metals_avanza_helpers,
data/metals_history_fetch, data/metals_llm, data/metals_warrant_refresh,
data/oil_loop, data/oil_swing_trader, data/oil_swing_config,
data/oil_warrant_refresh, data/crypto_loop, data/crypto_swing_trader,
data/crypto_swing_config, data/crypto_warrant_refresh, data/fish_engine,
data/fish_monitor, portfolio/exit_optimizer, portfolio/price_targets,
portfolio/fin_snipe, portfolio/fin_snipe_manager, portfolio/fin_fish,
portfolio/grid_fisher, portfolio/oil_grid_signal.
Agent: pr-review-toolkit:code-reviewer. → 04_metals_core.md

### 5. avanza-api — auth, orders, account, resilient page
avanza_session, avanza_client, avanza_orders, avanza_control,
avanza_tracker, avanza_order_lock, avanza_resilient_page, avanza_account_check.
Agent: caveman:cavecrew-reviewer (tight scope). → 05_avanza_api.md

### 6. signals-modules — 58 plugin files in portfolio/signals/
All portfolio/signals/*.py. Focus active first; check disabled for leakage.
Agent: pr-review-toolkit:code-reviewer. → 06_signals_modules.md

### 7. data-external — third-party fetchers
data_collector, fear_greed, sentiment, social_sentiment, alpha_vantage,
futures_data, onchain_data, fx_rates, crypto_macro_data, funding_rate,
news_keywords, earnings_calendar, econ_dates, fomc_dates, price_source,
http_retry, microstructure_state, metals_orderbook, metals_cross_assets.
Agent: pr-review-toolkit:code-reviewer. → 07_data_external.md

### 8. infrastructure — atomic I/O, locks, telegram, journals, dashboard
file_utils, health, shared_state, process_lock, subprocess_utils, api_utils,
config_validator, telegram_notifications, telegram_poller, message_store,
journal, dashboard/*.
Agent: pr-review-toolkit:code-reviewer. → 08_infrastructure.md

## My own independent pass (parallel with subagents)
- Atomic I/O violations
- Concurrency (ThreadPoolExecutor + shared dict mutations)
- Layer 2 subprocess pitfalls
- Voting math + HOLD dilution
- Stop-loss API misuse
- Float money math
- Datetime tz-naive vs tz-aware
- Bare except: catches that swallow errors
- File-handle leaks
→ 00_independent_pass.md

## Synthesis
Read each, drop hallucinations, dedupe, re-rank P1/P2/P3.
→ SYNTHESIS.md

### 9. trading-bots — independent bot subpackages
portfolio/golddigger/* (10 files), portfolio/elongir/* (10 files),
portfolio/mstr_loop/* + strategies/ (15 files).
Agent: pr-review-toolkit:code-reviewer. → 09_trading_bots.md

### 10. supporting-modules — precomputes, prophecy, evolve, qwen3, etc.
bigbet, fin_evolve, qwen3_signal, qwen3_trader, prophecy, focus_analysis,
reflection, shadow_registry, sentiment_shadow_backfill,
decision_outcome_tracker, vector_memory, bert_sentiment, regime_alerts,
short_horizon, daily_digest, weekly_digest, digest, notification_text,
seasonality, seasonality_updater, *_precompute (6 files),
feature_normalizer, tickers, stats, analyze, backtester, data_refresh,
fish_monitor_smart, fish_instrument_finder, tinylora_trainer.
Agent: pr-review-toolkit:code-reviewer. → 10_supporting.md

## Premortem (from fresh agent)

**1. Hallucinated finding (signals-modules has 58 files; reviewer cites function that was renamed last week).**
Causal: agent reads file, recalls stale pattern, claims bug that current code already fixes. Fix worktree edits non-bug, introduces real bug.
**Hook applied:** Every finding cites `file:line` + 3-line code quote. Synthesis greps quotes against live file; non-matching dropped.

**2. Coverage gap — `portfolio/golddigger/`, `portfolio/elongir/`, `portfolio/mstr_loop/` missing from partitions.**
**Hook applied:** Added partitions 9 + 10. Inventory cross-check ran: `git ls-files` diffed against assigned files; 70 unassigned files placed in partitions 9-10.

**3. Stale code drift — main loop writes `data/agent_summary.json` etc every 60s.**
**Hook applied:** Reviewers explicitly forbidden from reading `data/*.json` and `data/*.jsonl`. Code only.

**4. Cross-subsystem blind spot — bug in seam between two partitions.**
**Hook applied:** Independent pass owns explicit seam checklist: (a) shared_state consumers, (b) journal writers vs dashboard readers, (c) trigger_state producers vs autonomous consumers.

**5. Subagent silent failure — agent returns success with empty output.**
**Hook applied:** Synthesis step asserts each report exists, ≥30 lines, contains "P1", "P2", "P3" or `(none)`. Failed reports re-dispatched, not silently synthesized.

**6. Prioritization error — fresh reviewer lacks grudges memory.**
**Hook applied:** Every dispatch prompt embeds the P1 auto-promote list: stop-loss API endpoint, config.json commit, CLAUDECODE env leak, non-atomic JSON write, claude -p exit-0-on-auth-fail, MIN_VOTERS bypass.

