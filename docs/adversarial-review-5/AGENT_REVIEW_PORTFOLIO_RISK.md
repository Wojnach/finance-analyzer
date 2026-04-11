# Agent Review: portfolio-risk — Round 5 (2026-04-11)

**Agent**: feature-dev:code-reviewer
**Files reviewed**: 15 (portfolio_mgr.py, portfolio_validator.py, risk_management.py,
trade_guards.py, trade_risk_classifier.py, trade_validation.py, equity_curve.py,
monte_carlo.py, monte_carlo_risk.py, kelly_sizing.py, kelly_metals.py,
circuit_breaker.py, cost_model.py, exposure_coach.py, warrant_portfolio.py)

---

## Findings (10 total: 2 P0, 5 P1, 2 P2, 1 P3)

### P0 — Critical

**PR-R5-1** check_drawdown() never called from production (CONFIRMED from SO-1)
- risk_management.py — 20% drawdown circuit breaker completely inoperative
- Zero callers outside tests. A-PR-2 streaming fix is wasted.

**PR-R5-2** record_trade() never called from production (CONFIRMED from IR-2)
- trade_guards.py:177 — All 3 overtrading guards permanently blind
- Cooldown state never written, should_block_trade always returns False

### P1 — Important (NEW findings not in independent review)

**PR-R5-3** [NEW] warrant_portfolio.py:218 — Average-in BUY doesn't update underlying_entry_price_usd
- Only entry_price_sek updated with weighted average
- warrant_pnl() computes P&L against stale original underlying price
- 1% underlying error → 5% reported P&L error on 5x leverage warrants

**PR-R5-4** [NEW] trade_validation.py:32 — Default min order 500 SEK vs Avanza floor 1000 SEK
- validate_trade() approves orders 500-999 SEK that Avanza will reject
- kelly_sizing.py:290 and kelly_metals.py:44 also use 500 SEK
- Should be 1000 SEK per CLAUDE.md rules

**PR-R5-5** [NEW] risk_management.py:791 — check_atr_stop_proximity() called with "CHECK" sentinel
- Works by accident (guard checks action == "HOLD", "CHECK" passes through)
- Fragile if guard logic is ever changed to whitelist approach

**PR-R5-6** [NEW] equity_curve.py:233 — Sharpe ratio guard uses wrong units
- daily_vol in percentage units guards decimal-unit Sharpe computation
- Guard is always true (dead code), Sharpe computation itself is correct
- Double-computation of std deviation introduces maintenance risk

**PR-R5-7** [NEW] kelly_sizing.py:95 — Fee asymmetry biases Kelly win rate
- BUY total_sek includes fee (inflated cost), SELL total_sek is post-fee (deflated)
- Understates wins, overstates losses → Kelly fraction is conservative (safe but suboptimal)

### P2 — Lower Priority

**PR-R5-8** [NEW] monte_carlo_risk.py:211 — shares != 0 allows negative shares
- Negative-share positions (corruption) would invert P&L sign in VaR simulation
- Fix: change to > 0

**PR-R5-9** [NEW] kelly_metals.py:215 — Near-zero avg_loss produces 95% position sizing
- Tiny avg_loss from noisy DB → Kelly fraction explodes, clamped to 0.95
- Fix: add floor avg_loss = max(avg_loss, 0.5)

### P3

**PR-R5-10** warrant_portfolio.py:80 — Falsy numeric guards fail on price == 0.0
- `not current_underlying_usd` is True when price == 0.0 (degenerate but valid)
- Fix: use explicit `is None` checks

---

## Regression Check

| Prior Fix | Status |
|-----------|--------|
| A-PR-2 (streaming max) | CORRECT but dead code (check_drawdown never called) |
| A-PR-3 (raw json.load) | CORRECT — uses load_json() |
| PR-R4-4 (should_block_trade severity) | CORRECT but insufficient (record_trade never called) |
