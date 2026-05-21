# /fgl Adversarial Review Plan — 2026-05-21

## Goal
Full adversarial review of `finance-analyzer` at `b39f5b3e`. Partition codebase into 8 subsystems, run parallel fresh-context Claude Code review subagents alongside a lead independent pass. Cross-critique. Ship synthesis doc only — no code changes.

## Baseline SHA
`b39f5b3e68b759472fe6f648c8bf1baa1195aa78` (main, 2026-05-21)

## Subsystem Partition

| # | Name | Scope | Reviewer Agent |
|---|------|-------|----------------|
| 1 | signals-core | `signal_engine.py`, `signal_registry.py`, `signal_weights.py`, `signal_utils.py`, `signal_history.py`, `signal_state_since.py`, `signal_postmortem.py`, `signal_weight_optimizer.py`, `train_signal_weights.py`, `ic_computation.py`, `accuracy_stats.py`, `outcome_tracker.py`, `forecast_accuracy.py`, `meta_learner.py` | pr-review-toolkit:code-reviewer |
| 2 | orchestration | `main.py`, `agent_invocation.py`, `trigger.py`, `market_timing.py`, `claude_gate.py`, `autonomous.py`, `reporting.py`, `journal.py`, `journal_index.py`, `loop_health.py`, `health.py`, `crypto_scheduler.py`, `circuit_breaker.py` | pr-review-toolkit:code-reviewer |
| 3 | portfolio-risk | `portfolio_mgr.py`, `portfolio_validator.py`, `trade_guards.py`, `risk_management.py`, `equity_curve.py`, `monte_carlo.py`, `monte_carlo_risk.py`, `exit_optimizer.py`, `price_targets.py`, `warrant_portfolio.py`, `cost_model.py`, `trade_risk_classifier.py` | pr-review-toolkit:code-reviewer |
| 4 | metals-core | `data/metals_loop.py`, `portfolio/metals_*.py`, `portfolio/fin_snipe*.py`, `portfolio/grid_fisher.py`, `portfolio/silver_precompute.py`, `portfolio/gold_precompute.py`, `portfolio/orb_*.py`, `portfolio/iskbets.py`, `portfolio/fin_fish.py`, `portfolio/fish_*.py`, `portfolio/metals_ladder.py` | pr-review-toolkit:code-reviewer |
| 5 | avanza-api | `avanza_session.py`, `avanza_orders.py`, `avanza_client.py`, `avanza_account_check.py`, `avanza_control.py`, `avanza_resilient_page.py`, `avanza_tracker.py`, `avanza_order_lock.py` | caveman:cavecrew-reviewer |
| 6 | signals-modules | `portfolio/signals/*.py` (65 modules) | pr-review-toolkit:code-reviewer |
| 7 | data-external | `data_collector.py`, `fear_greed.py`, `sentiment.py`, `alpha_vantage.py`, `futures_data.py`, `onchain_data.py`, `fx_rates.py`, `news_keywords.py`, `earnings_calendar.py`, `econ_dates.py`, `funding_rate.py`, `crypto_macro_data.py`, `fomc_dates.py`, `bert_sentiment.py` | pr-review-toolkit:code-reviewer |
| 8 | infrastructure | `file_utils.py`, `http_retry.py`, `gpu_gate.py`, `shared_state.py`, `message_throttle.py`, `message_store.py`, `api_utils.py`, `logging_config.py`, `feature_normalizer.py`, `telegram_*.py`, `dashboard/app.py`, `data_refresh.py` | pr-review-toolkit:code-reviewer |

## Workflow

1. Create worktree `worktrees/review-2026-05-21` at `b39f5b3e`. Single shared worktree (read-only review, no per-subsystem branches needed because reviews are docs-only).
2. Spawn 8 subagents in single message (parallel). Each receives:
   - The subsystem file list
   - Severity bands (P0 incident-level, P1 must-fix, P2 should-fix, P3 nit)
   - Output path: `docs/reviews/2026-05-21/<subsystem>.md`
   - Constraint: read-only, no code edits
   - Format: `path:line: <severity>: <problem>. <fix>.`
3. While subagents run, lead writes independent pass at `docs/reviews/2026-05-21/lead-review.md`.
4. Collect, dedup, cross-critique. Write `docs/reviews/2026-05-21/synthesis.md` with:
   - Top 10 P0/P1 findings (severity-ranked, with cross-reviewer consensus)
   - Unique findings per reviewer
   - Findings the lead missed
   - Findings subagents missed but lead caught
   - Backlog feed (P2/P3 -> `docs/IMPROVEMENT_BACKLOG.md` deltas)
5. Commit all docs to `main`. Push via `cmd.exe /c "cd /d Q:\finance-analyzer && git push"`.
6. Clean up worktree.

## What Could Break

- Subagents may time out or produce empty findings. Mitigation: brief them with concrete file lists + line-budget guidance, run in background, collect what returns.
- Lead pass may drift toward subagent framing if I read subagent output first. Mitigation: lead pass written before collecting subagent results.
- Cavecrew-reviewer constrained to tight diffs — used only for avanza-api (smallest scope). Larger scopes use pr-review-toolkit:code-reviewer.

## Premortem

Skipped — this is review-only, no code shipped, no Layer 2 behavior change, no atomic-I/O surface. The "what could break" section covers reviewer-process risks. Rationale documented for the protocol auditor.

## Execution Order

1. Commit this plan.
2. Create worktree.
3. Spawn 8 subagents (single message, background).
4. Lead independent review pass (foreground, all 8 subsystems).
5. Collect subagent results.
6. Synthesize.
7. Commit `docs/reviews/2026-05-21/*` + `docs/PLAN.md` to main.
8. Push, clean up worktree.
