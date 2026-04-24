# Dual Adversarial Review #7 — Plan (2026-04-24)

Protocol: `/fgl` → `docs/GUIDELINES.md`. Prior review: `ADVERSARIAL_REVIEW_2026-04-17.md`.

## Scope

Full codebase adversarial review across 8 subsystems. Focus on:
- Bugs introduced in the ~30 commits since 2026-04-17
- Regressions from prior fixes (CR-1 through CR-7 from review #6)
- Cross-cutting concerns missed by per-subsystem reviews
- Trading-specific correctness (price, position sizing, order logic)

## Subsystems

1. **signals-core** — signal_engine, accuracy_stats, signal_db, outcome_tracker, etc.
2. **orchestration** — main.py, trigger, agent_invocation, loop_contract, market_timing
3. **portfolio-risk** — portfolio_mgr, risk_management, trade_guards, monte_carlo
4. **metals-core** — metals_precompute, fin_fish, fin_snipe_manager, exit_optimizer, iskbets
5. **avanza-api** — avanza_session, avanza_orders, portfolio/avanza/ package
6. **signals-modules** — 38 signal plugins in portfolio/signals/
7. **data-external** — data_collector, sentiment, futures_data, onchain_data, etc.
8. **infrastructure** — file_utils, shared_state, health, telegram, dashboard, bots

## Methodology

1. Launch 8 parallel code-reviewer agents (one per subsystem)
2. Write independent lead-reviewer findings on critical cross-cutting files
3. Cross-critique in both directions (agent vs lead, lead vs agent)
4. Classify: CONFIRMED / FALSE POSITIVE / DOWNGRADED / NOVEL
5. Write synthesis doc with prioritized findings

## Deliverable

`docs/ADVERSARIAL_REVIEW_2026-04-24.md` — committed to main, pushed.
