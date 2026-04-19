# Agent Review: portfolio-risk

## P1 Findings
1. **Drawdown circuit breaker never called** for Patient/Bold portfolios — function exists but not wired into main path (risk_management.py:86)
2. **trade_guards.record_trade never called** for main portfolios — all guards permanently bypassed (trade_guards.py:177)

## P2 Findings
1. **record_trade read-modify-write race** — no threading lock (trade_guards.py:189-235)
2. **portfolio_mgr non-reentrant Lock** — potential deadlock if mutate_fn calls save_state (portfolio_mgr.py:107-158)
3. **equity_curve gross P&L** — fees excluded, overstates profitability (equity_curve.py:514)
4. **Monte Carlo stop_price anchored to current** not entry price (monte_carlo.py:292)
5. **check_drawdown blind on feed failure** — returns breached=False when can't compute value (risk_management.py:115-139)

## P3 Findings
1. trade_validation min_order_sek defaults to 500, should be 1000
2. equity_curve naive datetime mixing
3. check_atr_stop_proximity "CHECK" sentinel undocumented
4. monte_carlo_risk correlation uses oldest observations instead of recent
5. Two duplicate portfolio_value implementations with different fallback behavior
6. trade_risk_classifier double-counts trending-down regime penalty
