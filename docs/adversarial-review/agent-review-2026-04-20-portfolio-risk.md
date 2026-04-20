# Agent Review: portfolio-risk (2026-04-20)

## P1 Critical
1. **Drawdown circuit breaker is STILL dead code** — `check_drawdown()` has 16 tests, 0 production callers. System trades through unlimited drawdown.
2. **`record_trade()` STILL never called** — `trade_guard_state.json` doesn't exist. Overtrading guards are non-functional (C4 warning logs this explicitly).
3. **`should_block_trade()` STILL dead code** — No production caller exists.
4. **No programmatic gate between Layer 2 and portfolio state** — LLM directly edits JSON files. No pre_trade_hook, no validation. Hallucination → bad state persisted.
5. **`trade_validation.py` and `trade_risk_classifier.py` entirely orphaned** — Never imported from production.

## P2 High
1. Kelly sizing is advisory only (playbook hardcodes 15%/30%, ignores Kelly)
2. Monte Carlo VaR is report-only (no threshold blocks trades)
3. Risk flags are informational, never blocking
4. check_drawdown has stale-data blindspot (falls back to cash-only when prices unavailable)

## P3 Medium
1. circuit_breaker.py name misleading (API retry CB, not portfolio CB)
2. GoldDigger has working risk that Patient/Bold lacks (proves pattern works)
3. Kelly overflow at very low cert_loss_frac (capped at 0.95 but advisory-only)

## Actual Risk Enforcement Path (Patient/Bold)
What ACTUALLY runs: Signal consensus, playbook suggestions (natural language to LLM)
What's DEAD CODE: check_drawdown, should_block_trade, record_trade, validate_trade, classify_trade_risk, Kelly, MC VaR, risk flags
What prevents 50% drawdown: **NOTHING**
