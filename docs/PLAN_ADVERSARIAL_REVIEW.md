# PLAN — Dual Adversarial Review of finance-analyzer

**Created:** 2026-04-05
**Author:** Claude (Opus 4.6 1M)
**Scope:** Adversarial review of the entire finance-analyzer trading system, performed
by Codex (GPT-5 family) AND Claude independently, with cross-critique between the two
reviewers.

## Goal

Stress-test the architecture, design choices, and implementation of the finance-analyzer
trading system. This is a **review-only** task — no code changes. The output is a set of
review documents that a human operator can act on (or not).

## Why dual review

A single reviewer has blind spots. By running Codex and Claude independently against
the same code and then having each critique the other's findings, we get:

- **Coverage**: each reviewer catches things the other misses.
- **Calibration**: agreed findings have higher confidence than solo findings.
- **Adversarial discipline**: the meta-review round forces each reviewer to defend
  findings or discard false positives.
- **Different lenses**: Codex reviews the git-diff the companion tool feeds it; Claude
  reads files directly and can cross-reference conversation memory (past incidents,
  known pain points, the user's documented trading rules).

## Technical constraint: codex reviews diffs, not arbitrary code

`/codex:adversarial-review` runs against a git diff — either `working-tree` or a
`branch ... base` diff. It does not have a "whole codebase" mode. To make codex review
an arbitrary set of files, we create a synthetic diff:

1. Create a worktree off main: `worktrees/adversarial-review`
2. Inside the worktree, create an orphan branch `empty-baseline` with zero tracked files
3. For each subsystem, create a branch `review/<subsys>` off `empty-baseline` that
   contains only that subsystem's files (copied from main)
4. Run `/codex:adversarial-review --scope branch --base empty-baseline` from the
   subsystem branch — codex now sees the whole subsystem as "added files"

This is clean: each codex review is bounded, topical, and reproducible. Cleanup is
`git worktree remove && git branch -D <branches>`.

## Partitioning — 8 subsystems

Chosen to (a) bound each diff so codex doesn't choke, (b) map to architectural layers,
(c) co-locate files that share invariants so reviewers can see coupling.

### 1. signals-core (voting + weighting + accuracy)
Files: `signal_engine.py`, `signal_registry.py`, `signal_utils.py`, `accuracy_stats.py`,
`outcome_tracker.py`, `ticker_accuracy.py`, `signal_weights.py`,
`signal_weight_optimizer.py`, `train_signal_weights.py`, `signal_postmortem.py`,
`signal_history.py`, `signal_db.py`.

Challenge: voting math, MIN_VOTERS rule, quorum gates, regime multipliers, 45%
accuracy gate, recency weighting, weight-by-accuracy coupling, force-HOLD logic,
whiplash risk.

### 2. orchestration (Layer 1 + Layer 2 + autonomous fallback)
Files: `main.py`, `agent_invocation.py`, `trigger.py`, `market_timing.py`,
`autonomous.py`, `reporting.py`, `claude_gate.py`, `loop_contract.py`,
`multi_agent_layer2.py`, `reflection.py`, `perception_gate.py`, `health.py`.

Challenge: 60s loop lifecycle, crash recovery math (10s→5min backoff), CLI subprocess
safety, trigger idempotence, DST handling, T1/T2/T3 tier escalation, autonomous
fallback rules correctness, health state truthfulness, module-failure propagation.

### 3. portfolio-risk
Files: `portfolio_mgr.py`, `portfolio_validator.py`, `trade_guards.py`,
`trade_validation.py`, `risk_management.py`, `equity_curve.py`, `monte_carlo.py`,
`monte_carlo_risk.py`, `kelly_sizing.py`, `cost_model.py`, `circuit_breaker.py`,
`trade_risk_classifier.py`, `warrant_portfolio.py`.

Challenge: atomic state I/O under concurrency, guard ordering, drawdown circuit
breaker math, SEK/USD conversion correctness, round-trip P&L, concentration limits,
ATR stops, Kelly sizing edge cases, Monte Carlo GBM assumptions, t-copula VaR/CVaR.

### 4. metals-core (metals loop + microstructure + local bots)
Files: `data/metals_loop.py` (5,261 LOC — largest file in repo),
`portfolio/metals_cross_assets.py`, `portfolio/metals_orderbook.py`,
`portfolio/metals_ladder.py`, `portfolio/metals_precompute.py`,
`portfolio/microstructure.py`, `portfolio/microstructure_state.py`,
`portfolio/exit_optimizer.py`, `portfolio/price_targets.py`, `portfolio/fin_snipe.py`,
`portfolio/fin_snipe_manager.py`, `portfolio/fin_fish.py`, `portfolio/iskbets.py`,
`portfolio/orb_predictor.py`, `portfolio/orb_backtest.py`, `portfolio/orb_postmortem.py`.

Challenge: orderbook snapshot accumulation, persisted OFI state correctness, fast-tick
thread safety, silver embedded monitor coordination, exit optimizer probability model,
price target structural levels, fin_snipe ladder math, orb predictor look-ahead bias,
live Avanza position sync, DST-aware closing-time logic.

### 5. avanza-api (authentication + order flow)
Files: `portfolio/avanza_session.py`, `portfolio/avanza_orders.py`,
`portfolio/avanza_client.py`, `portfolio/avanza_control.py`,
`portfolio/avanza_tracker.py`, `portfolio/avanza/*.py` (11 files in unified submodule:
`__init__`, `account`, `auth`, `client`, `market_data`, `scanner`, `search`,
`streaming`, `tick_rules`, `trading`, `types`).

Challenge: BankID session expiry and reauth paths, volume constraint enforcement
(position size vs sell+stop-loss volumes), stop-loss API endpoint correctness
(the `/_api/trading/stoploss/new` vs regular order API incident from Mar 3),
todayClosingTime DST handling, order idempotence, streaming reconnect safety,
tick-rule rounding correctness, credential handling.

### 6. signals-modules (21 plugin modules)
Files: all of `portfolio/signals/*.py`: `trend.py`, `momentum.py`, `volume_flow.py`,
`volatility.py`, `candlestick.py`, `structure.py`, `fibonacci.py`, `smart_money.py`,
`oscillators.py`, `heikin_ashi.py`, `mean_reversion.py`, `calendar_seasonal.py`,
`macro_regime.py`, `momentum_factors.py`, `news_event.py`, `econ_calendar.py`,
`forecast.py`, `claude_fundamental.py`, `futures_flow.py`, `orderbook_flow.py`,
`metals_cross_asset.py`, `crypto_macro.py`.

Challenge: look-ahead bias in window logic, NaN handling, hardcoded dates (FOMC/CPI
lists), upstream data availability assumptions, per-asset-class applicability gates,
weight sanity, false-positive rate under ranging regimes, signal correlation
(redundancy between trend/momentum/heikin), model-dependent signals (forecast,
claude_fundamental) failure modes.

### 7. data-external (market data + sentiment + news)
Files: `portfolio/data_collector.py`, `portfolio/sentiment.py`,
`portfolio/fear_greed.py`, `portfolio/alpha_vantage.py`, `portfolio/futures_data.py`,
`portfolio/onchain_data.py`, `portfolio/fx_rates.py`, `portfolio/news_keywords.py`,
`portfolio/crypto_macro_data.py`, `portfolio/earnings_calendar.py`,
`portfolio/social_sentiment.py`, `portfolio/macro_context.py`,
`portfolio/market_health.py`.

Challenge: provider-specific retry/backoff, rate-limit accounting (Alpha Vantage 25/day,
NewsAPI 100/day, BGeometrics 15/day), timezone normalization across sources,
Binance vs Alpaca vs yfinance data drift, sentiment keyword-weighting correctness,
FX caching staleness, on-chain metric staleness.

### 8. infrastructure (I/O, locking, notifications)
Files: `portfolio/file_utils.py`, `portfolio/http_retry.py`, `portfolio/shared_state.py`,
`portfolio/gpu_gate.py`, `portfolio/process_lock.py`, `portfolio/journal.py`,
`portfolio/journal_index.py`, `portfolio/prophecy.py`,
`portfolio/telegram_notifications.py`, `portfolio/telegram_poller.py`,
`portfolio/message_store.py`, `portfolio/message_throttle.py`,
`portfolio/subprocess_utils.py`, `portfolio/logging_config.py`,
`portfolio/log_rotation.py`, `portfolio/backup.py`, `portfolio/config_validator.py`,
`portfolio/api_utils.py`, `portfolio/alert_budget.py`.

Challenge: atomic_write_json correctness (fsync, rename atomicity, partial writes),
append_jsonl race handling, subprocess timeout paths, GPU lock fairness, process lock
staleness detection, journal schema drift, Telegram rate-limit and delivery retry,
message dedup, config secret exposure, log rotation under concurrent writers.

### Out of scope (intentionally)

- `backtester.py`, `*_precompute.py`, `*_backtest.py` — offline tools, low operational
  risk.
- `portfolio/golddigger/*`, `portfolio/elongir/*` — bots with their own scope.
- `data/*.py` except `metals_loop.py` — scripts, not production runtime.
- `tests/` — reviewing production code, not tests.
- Dashboard (`dashboard/*`) — serves data, doesn't decide.

## Execution flow

### Phase A — Setup
1. Create worktree `worktrees/adversarial-review` off main.
2. In worktree, create orphan `empty-baseline` branch with zero tracked files.
3. For each of 8 subsystems, create `review/<subsys>` branch off `empty-baseline`,
   populate with files copied from main, commit.
4. Commit this plan to main.

### Phase B — Parallel execution
5. From within the worktree on each `review/<subsys>` branch, launch
   `/codex:adversarial-review --scope branch --base empty-baseline "<focus>"` in
   background (up to 8 background jobs).
6. In parallel, Claude reads each subsystem from main and writes findings to
   `docs/ADVERSARIAL_REVIEW_CLAUDE.md` — one section per subsystem.

### Phase C — Collect and critique
7. Consolidate codex output into `docs/ADVERSARIAL_REVIEW_CODEX.md`.
8. Claude meta-reviews Codex → `docs/META_REVIEW_CLAUDE_ON_CODEX.md`.
9. Codex meta-reviews Claude via `/codex:task` → `docs/META_REVIEW_CODEX_ON_CLAUDE.md`.

### Phase D — Synthesis and ship
10. Write `docs/ADVERSARIAL_REVIEW_SYNTHESIS.md` — consolidated findings.
11. Commit, merge to main, push via Windows git.
12. Delete review worktree and all `empty-baseline` + `review/*` branches.

## Deliverables

| Path | Content |
|------|---------|
| `docs/PLAN_ADVERSARIAL_REVIEW.md` | This plan |
| `docs/ADVERSARIAL_REVIEW_CODEX.md` | Codex's verbatim findings per subsystem |
| `docs/ADVERSARIAL_REVIEW_CLAUDE.md` | Claude's findings per subsystem |
| `docs/META_REVIEW_CLAUDE_ON_CODEX.md` | Claude critiquing Codex |
| `docs/META_REVIEW_CODEX_ON_CLAUDE.md` | Codex critiquing Claude |
| `docs/ADVERSARIAL_REVIEW_SYNTHESIS.md` | Final consolidated findings + action list |

## Risks & mitigations

- **Context overflow on metals-core (metals_loop.py is 5,261 LOC).** If the single
  job fails, split into metals-loop-a and metals-loop-b.
- **Resource contention across 8 background codex jobs.** Monitor via `/codex:status`;
  stagger if needed.
- **False positives inflate the review.** The meta-review round exists specifically
  to classify these.
- **Stale snapshot.** Capture HEAD SHA at start; note any drift at the end.
- **Reviewer bias cross-contamination.** Claude writes its review BEFORE reading
  any Codex output.

## Success criteria

- All 8 subsystems reviewed by both reviewers.
- Cross-critique completed in both directions.
- Synthesis identifies top consolidated issues with file paths, severity, reviewer
  agreement, and rationale.
- All docs committed and pushed; no dangling worktrees/branches.

## What this review is NOT

Not a style pass, not a test-coverage audit, not a performance review, not a security
audit (though security will be flagged if encountered), not a refactoring plan. The
framing is: **are the current design choices the right ones, and where is this system
most likely to fail under real trading conditions?**
