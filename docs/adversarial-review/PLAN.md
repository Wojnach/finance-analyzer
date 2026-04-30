# Adversarial Review #9 — Plan (2026-04-30)

## Objective
Full dual adversarial code review: 8 subsystems, parallel agent reviewers + independent manual review, cross-critique, synthesis. **Focus on code changed since 2026-04-23** (crypto swing trader, new signals, trade guard decay, bias detector, accuracy audit tooling) plus re-verification of prior unfixed P0/P1s.

## 8 Subsystems

| # | Subsystem | Key files | ~LOC | Changes since last review |
|---|-----------|-----------|------|---------------------------|
| 1 | signals-core | signal_engine, signal_registry, accuracy_stats, outcome_tracker, signal_weights, ticker_accuracy, accuracy_degradation | ~4K | +247 lines signal_engine (macro overlay, rescue), bias detector tune, voter quorum fix |
| 2 | orchestration | main, agent_invocation, trigger, market_timing, autonomous, multi_agent_layer2, loop_contract, crypto_scheduler | ~4K | crypto loop added, TOCTOU lock fix |
| 3 | portfolio-risk | portfolio_mgr, trade_guards, risk_management, equity_curve, monte_carlo*, circuit_breaker | ~3K | trade guard time-based decay (+60 lines), risk_management +54 lines |
| 4 | metals-core | metals_precompute, fin_fish, fin_snipe*, fish_monitor_smart, exit_optimizer, price_targets, orb_predictor, microstructure* | ~4K | minor fish_monitor changes |
| 5 | avanza-api | avanza_session, avanza_orders, avanza_client, avanza_control, avanza_order_lock, avanza_resilient_page, portfolio/avanza/* | ~3K | no major changes |
| 6 | signals-modules | portfolio/signals/*.py (45 modules) | ~8K | +vol_ratio_regime, +drift_regime_gate, +crypto_cross_asset, claude_fundamental +149, fibonacci disabled |
| 7 | data-external | data_collector, sentiment, futures_data, onchain_data, fx_rates, *_precompute, news_keywords, bert_sentiment, ministral_signal, qwen3_signal | ~4K | sentiment +217 lines, mstr_precompute +281, news_keywords +91, crypto_precompute new |
| 8 | infrastructure | file_utils, shared_state, telegram_*, reporting, dashboard, golddigger, elongir, digest, econ_dates | ~5K | reporting +35, dashboard crypto endpoints, tunnel scripts |

## Review Criteria
1. Bugs — logic errors, race conditions, off-by-one, dead code paths
2. Security — credential leaks, injection, unsafe deserialization
3. Reliability — silent failures, missing retries, crash paths
4. Data integrity — non-atomic writes, stale reads, corruption
5. Performance — unnecessary I/O, O(n²), memory leaks, thread contention
6. Architecture — coupling, god functions, circular deps
7. Correctness — signal math, wrong formulas, timezone bugs
8. **Prior P0/P1 re-verification** — check if prior findings were actually fixed

## Execution
1. ✅ Commit this plan
2. Launch 8 parallel code-reviewer agents (adversarial stance)
3. Simultaneously write independent manual review (read key files, focus on new code)
4. Collect agent results → docs/adversarial-review/agent-{nn}-{subsystem}-2026-04-30.md
5. Cross-critique both directions
6. Write SYNTHESIS-2026-04-30.md
7. Commit all, merge main, push, clean up
