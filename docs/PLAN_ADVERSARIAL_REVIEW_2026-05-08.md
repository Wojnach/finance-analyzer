# Dual Adversarial Review — 2026-05-08

## Goal

Stress-test the `finance-analyzer` codebase via two independent adversarial reviewers
(Codex GPT-5 family + Claude Opus 4.7), then cross-critique to filter false positives
and surface high-confidence findings.

## 8-Subsystem Partition

| # | Subsystem | Branch | Files |
|---|-----------|--------|-------|
| 1 | signals-core | `review/2026-05-08-signals-core` | signal_engine, signal_registry, signal_utils, signal_weights, signal_weight_optimizer, signal_db, signal_history, signal_postmortem, accuracy_stats, accuracy_degradation, ticker_accuracy, forecast_accuracy, outcome_tracker |
| 2 | orchestration | `review/2026-05-08-orchestration` | main, agent_invocation, trigger, market_timing, autonomous, claude_gate, config_validator, tickers, session_calendar, loop_contract, multi_agent_layer2, reflection, perception_gate, health, reporting |
| 3 | portfolio-risk | `review/2026-05-08-portfolio-risk` | portfolio_mgr, portfolio_validator, risk_management, trade_guards, trade_validation, trade_risk_classifier, equity_curve, monte_carlo, monte_carlo_risk, cost_model, circuit_breaker, kelly_sizing, warrant_portfolio |
| 4 | metals-core | `review/2026-05-08-metals-core` | data/metals_loop, metals_cross_assets, metals_orderbook, metals_ladder, metals_precompute, microstructure, microstructure_state, exit_optimizer, price_targets, fin_snipe, fin_snipe_manager, fin_fish, iskbets, orb_predictor, orb_postmortem, silver_precompute |
| 5 | avanza-api | `review/2026-05-08-avanza-api` | avanza_session, avanza_orders, avanza_order_lock, avanza_resilient_page, avanza_tracker, avanza_control, avanza_client, portfolio/avanza/* |
| 6 | signals-modules | `review/2026-05-08-signals-modules` | portfolio/signals/* (full directory) |
| 7 | data-external | `review/2026-05-08-data-external` | data_collector, sentiment, fear_greed, alpha_vantage, futures_data, onchain_data, fx_rates, news_keywords, crypto_macro_data, earnings_calendar, social_sentiment, macro_context, market_health, bert_sentiment |
| 8 | infrastructure | `review/2026-05-08-infrastructure` | file_utils, http_retry, shared_state, gpu_gate, process_lock, journal, journal_index, prophecy, telegram_notifications, telegram_poller, message_store, message_throttle, subprocess_utils, logging_config, log_rotation, api_utils, alert_budget, dashboard/app |

## Empty-baseline trick

`codex review` runs on a git diff. To make codex review whole subsystems, each
subsystem branch is built off an orphan `empty-baseline` branch with only that
subsystem's files. `codex review --base empty-baseline` then sees the entire
subsystem as "added files" — bounded, topical, reproducible.

## Severity scale

- **P0** Data loss, money loss, security vulnerability, silent failure hiding real problems.
- **P1** Logic errors causing wrong trades, race conditions, unhandled edge cases on prod paths.
- **P2** Code quality, missing validation, dead code, performance issues.
- **P3** Style, docstring gaps, minor cleanups.

## Execution flow

1. Worktree `Q:/finance-analyzer/.worktrees/adversarial-review-2026-05-08` exists. Empty
   baseline + 8 subsystem branches built.
2. Spawn 8 background `codex review --base empty-baseline` jobs (one per subsystem branch).
3. While codex runs, Claude reads each subsystem from main and writes
   `docs/ADVERSARIAL_REVIEW_2026-05-08_CLAUDE.md` — independent findings.
4. Collect codex output → `docs/ADVERSARIAL_REVIEW_2026-05-08_CODEX.md`.
5. Cross-critique:
   - `docs/META_REVIEW_2026-05-08_CLAUDE_ON_CODEX.md` — Claude critiques codex.
   - `docs/META_REVIEW_2026-05-08_CODEX_ON_CLAUDE.md` — codex critiques Claude.
6. Synthesis → `docs/ADVERSARIAL_REVIEW_2026-05-08_SYNTHESIS.md`.
7. Commit all docs to main, push via Windows git, delete branches + worktree.

## Deliverables

| Path | Purpose |
|------|---------|
| `docs/PLAN_ADVERSARIAL_REVIEW_2026-05-08.md` | This plan |
| `docs/ADVERSARIAL_REVIEW_2026-05-08_CLAUDE.md` | Claude's independent findings |
| `docs/ADVERSARIAL_REVIEW_2026-05-08_CODEX.md` | Codex's findings (concatenated) |
| `docs/META_REVIEW_2026-05-08_CLAUDE_ON_CODEX.md` | Claude critiquing codex |
| `docs/META_REVIEW_2026-05-08_CODEX_ON_CLAUDE.md` | Codex critiquing Claude |
| `docs/ADVERSARIAL_REVIEW_2026-05-08_SYNTHESIS.md` | Final consolidated findings |

## Constraints

- Read-only review. No production code changes in this session.
- Claude writes its review *before* reading any codex output.
- All 8 codex jobs must complete; partial results documented if any time out.
- Synthesis ranks by severity × reviewer-agreement.

## Risks

- metals-core (`data/metals_loop.py` ~5K LOC) may exceed codex context. Mitigation:
  if it fails, split metals-loop-a (top half) / metals-loop-b (bottom half).
- 8 concurrent codex jobs may rate-limit. Mitigation: stagger spawns 30s apart.
- Worktree dirty after subsystem builds — won't be merged, only docs reach main.

## Success criteria

All 8 subsystems reviewed by both reviewers, cross-critique done in both
directions, synthesis identifies top consolidated issues with file paths,
severity, and reviewer agreement. All docs committed/pushed; no dangling
worktrees/branches.
