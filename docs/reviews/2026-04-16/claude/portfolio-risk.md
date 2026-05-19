# Adversarial Review — portfolio-risk (Claude-independent)

## Executive Summary
Live trading system with 500K SEK per strategy. 13 issues ranging from P0 (atomicity/money math) to P3 (documentation).

## Critical Findings

### P0: FX Conversion Double-Application in Portfolio Valuation
**File:** `portfolio/portfolio_mgr.py:171`, called from `risk_management.py:119`
`total += shares * price * fx_rate` — if `prices_usd` is pre-converted (or FX applied twice up the call chain), portfolio value swings ~10–20% with SEK/USD ~10.5x. Affects drawdown circuit breaker threshold, Kelly sizing, P&L reporting.
**Fix:** rename `prices_usd` globally, assert `price_usd > 0`, document invariant "FX applied once, last."

### P0: Atomicity Failure — State Read vs Equity Curve Write Race
**File:** `risk_management.py:134` + `portfolio_mgr.py:152-157`
`update_state()` holds lock during RMW, but `log_portfolio_value()` appends to `portfolio_value_history.jsonl` OUTSIDE the lock. Crash between state save and history append → history peak lags portfolio state → drawdown math stale → circuit breaker reads stale peak → more trades allowed than should be.
**Fix:** write history BEFORE state, or dual-write in single atomic op. Alternatively compute peak from state (initial_value), use history as secondary.

### P1: Unbounded Kelly Sizing — No Fraction Cap Before 2.0 Leverage
**File:** `kelly_sizing.py:277-279`
`kelly_fraction()` clamps to `[0, 1]` (too loose). Full Kelly 0.50 → half 0.25 → 125K SEK (25% of 500K). Four-loss streak + accuracy drop to 45% = negative log-growth. 500 SEK min is checked AFTER half Kelly, not before.
**Fix:** cap full Kelly at 0.25 before halving. Add edge validation: if `win_prob * avg_win <= (1 - win_prob) * avg_loss`, return kelly_pct=0 with warning.

### P1: Monte Carlo GBM Drift vs t-Copula Tail Mismatch
**File:** `monte_carlo.py:60-89`, `monte_carlo_risk.py:38`
Individual paths are Gaussian; joint structure is t-copula (df=4, heavy tails). Individual tails can't produce the 0.18 lower-tail dependence the copula assumes. VaR underestimated during crashes.
**Fix:** Student-t marginals in GBM, OR document limitation + apply `var_stressed = var * 1.15` for df=4.

### P1: Sharpe/Sortino Annualization — N vs N-1 Divisor Mismatch
**File:** `equity_curve.py:225-248`
Sharpe uses N-1 divisor (line 240 `daily_std_dec`), but Sortino uses N (line 245 `sum/len`). 1% systematic bias in Sortino for small N. Strategy rankings (Patient vs Bold) could flip on early backtests.
**Fix:** N-1 everywhere. Document 252 assumption; crypto should use 365.25.

### P1: Stop Near Knock-Out Barrier — No Guard in monte_carlo.py
**File:** `risk_management.py:155-173` + `monte_carlo.py:292`
`stop_price = price * (1 - 2 * atr_pct / 100)` static. With ATR 8%, price 20, stop=18.4, barrier=17.5 → 17.8 spike = knockout (100% loss) despite stop placement intent. **Direct violation of user grudge #1.**
**Fix:** guard `stop_price_usd > barrier_price_usd * 1.02`; if fails, HOLD or reduce size.

### P1: Kelly Edge Ignores Transaction Costs
**File:** `kelly_sizing.py:260-274`
`avg_win`/`avg_loss` come from post-fee P&L, but `win_prob` from signal direction (pre-fee). Systematic 5–10% under-sizing.
**Fix:** document, or compute two Kellys (net for sizing, gross for reporting).

### P2: Concentration Limit Uses Cash% Not Portfolio%
**File:** `trade_validation.py:75-81`
`cash_pct = (order_value / cash_available) * 100` — existing holdings ignored. Single ticker can reach 100% via repeated <50% buys.
**Fix:** total exposure check `(holdings[ticker].value + order_value) / portfolio_value > 0.35 → reject`.

### P2: Circuit Breaker Volatility-Blind
**File:** `circuit_breaker.py:69-93`
Only watches drawdown. Vol jump from 5% → 25% with flat P&L → 0 drawdown → breaker doesn't fire. Risk quintupled but Kelly unchanged.
**Fix:** `adjusted_max_dd = max_dd * (baseline_vol / current_vol)`; reduce Kelly 25% if vol > 2x baseline.

### P2: Round-Trip Fee Allocation Ambiguity
**File:** `equity_curve.py:305-408`
Multi-leg partial exits: proportional fee allocation recomputes per leg but single SELL has single fee. 1–2% P&L error possible on scale-outs.
**Fix:** document fee assumption; assert `sum(proportional_fees_per_leg) ≈ original_fee_sek` ±0.1%.

### P2: VaR Semantic Ambiguity
**File:** `monte_carlo_risk.py`
`var_95_usd` — 5th percentile of loss or 95th percentile of upside? Not docstring'd.
**Fix:** rename `var_95_loss_usd`; add docstring "P(Loss < VaR_95) = 0.05".

### P3: trade_risk_classifier.py Cost Asymmetry
Rule-based but thresholds 0-3 LOW / 4-6 MED / 7+ HIGH may underweight HIGH (false neg = 20x loss).
**Fix:** lower HIGH threshold to 6 or add override.

### P3: Annualization Factor Undocumented for Crypto
`math.sqrt(252)` but crypto 24/7 → underreports Sharpe/Sortino by ~11%.
**Fix:** detect asset mix; use `math.sqrt(365.25)` when crypto >30%.

## Looked OK
1. **atomic_write_json** — tempfile + replace + fsync, solid.
2. **portfolio_mgr locks** — per-file RMW locks.
3. **warrant_portfolio P&L** — leverage multiplication correct.
4. **cost_model** — presets reasonable (10-40 bps slippage).
5. **trade_validation** — pass/fail logic correct apart from P2.

## Reviewer confidence
0.75
